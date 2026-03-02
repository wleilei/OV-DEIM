import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import hydra 
from omegaconf import DictConfig
import os

from config.coco.dinov3_coco_l import DINOv3LConfig as coco_l
from config.coco.dinov3_coco_m import DINOv3MConfig as coco_m
from config.coco.dinov3_coco_s import DINOv3SConfig as coco_s

from hydra.core.config_store import ConfigStore
import time
import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataloader.data_utils import (get_coco_pipelines, get_coco_train_dataset, get_coco_test_dataset, 
                                   train_collate, eval_collate, get_label2catid)
from model.backbone.dinov3_adapter import DINOv3STAs
from model.encoder.hybrid_encoder import HybridEncoder
from model.decoder.ovdeim_decoder import OVDEIMDecoder
from model.criterion.ovdeim_criterion import OVDEIMCriterion
from model.ovdeim import OVDEIM
from model.ovdeim_postprocessor import OVDEIMPostProcessor

from dist_tools.dist_utils import get_available_gpus, setup
from optim_tools.ema import ModelEMA
from optim_tools.warmup import WarmupConstantCosine
from optim_tools.utils import get_optim_params, log_param_groups_to_swanlab, count_parameters

cs = ConfigStore.instance()
cs.store(name="coco_l", node=coco_l)
cs.store(name="coco_m", node=coco_m)
cs.store(name="coco_s", node=coco_s)
hydra.initialize(config_path="config", version_base=None)

args = hydra.compose(config_name="coco_l")

def evaluate(model, postprocessor, test_dataloader, label2catid, coco_gt, device, epoch_or_step="pretrain"):
    """Evaluate model on test dataset and return AP metrics."""
    model.to(device)
    model.eval()
    predictions = []
    
    print(f"Starting evaluation on {device}...")
    
    try:
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                try:
                    data = batch['inputs'].to(device).to(torch.float32)
                    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch['data_samples'].items()}
                    outputs = model(data, targets)
                    preds = postprocessor(outputs, targets)
                    
                    for pred, img_id in zip(preds, targets['img_id']):
                        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                            predictions.append({
                                'image_id': int(img_id.item()) if torch.is_tensor(img_id) else int(img_id),
                                'category_id': label2catid[int(label.detach().cpu().item())],
                                'bbox': [round(x, 2) for x in box.tolist()],
                                'score': float(score.item())
                            })
                except Exception as e:
                    print(f"Error processing batch {idx}: {e}")
                    continue
        
        print(f"Total predictions generated: {len(predictions)}")
        
        metrics = None
        
        if len(predictions) > 0:
            # Save predictions to temporary file
            import json
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(predictions, f)
                temp_file = f.name
            
            try:
                # Load predictions and evaluate
                coco_dt = coco_gt.loadRes(temp_file)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                # Extract all COCO mAP metrics (12 standard metrics)
                metrics = {
                    "AP": coco_eval.stats[0],      # mAP@0.5:0.95
                    "AP50": coco_eval.stats[1],    # mAP@0.5
                    "AP75": coco_eval.stats[2],    # mAP@0.75
                    "APs": coco_eval.stats[3],     # mAP@0.5:0.95 (small)
                    "APm": coco_eval.stats[4],     # mAP@0.5:0.95 (medium)
                    "APl": coco_eval.stats[5],     # mAP@0.5:0.95 (large)
                    "AR1": coco_eval.stats[6],     # AR@1
                    "AR10": coco_eval.stats[7],    # AR@10
                    "AR100": coco_eval.stats[8],   # AR@100
                    "ARs": coco_eval.stats[9],     # AR@100 (small)
                    "ARm": coco_eval.stats[10],    # AR@100 (medium)
                    "ARl": coco_eval.stats[11],    # AR@100 (large)
                }
                
                print(f"\n{epoch_or_step} - AP: {metrics['AP']:.4f}, AP50: {metrics['AP50']:.4f}, AP75: {metrics['AP75']:.4f}")
                print("Detailed metrics:", metrics)
                    
            except Exception as e:
                print(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        else:
            print("Warning: No predictions generated for evaluation")
    except Exception as e:
        print(f"Error in evaluate function: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics

train_pipeline_1, train_pipeline_2, train_pipeline_syn, test_pipeline = get_coco_pipelines(size=args.data.img_scale, 
                                                                    num_texts=args.data.num_training_classes, 
                                                                    mixup_prob=args.data.mixup_prob,
                                                                    blank_text=args.data.blank_text,
                                                                    apply_moasic=args.data.apply_moasic,
                                                                    )

test_dataset = get_coco_test_dataset(
    data_test_root=args.data.data_val_root, 
    ann_test_file=args.data.ann_val_file, 
    cache_dir=args.data.cache_dir,
    pipeline_clean=test_pipeline
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1,
    collate_fn=eval_collate,
    num_workers=2
)

coco_gt = COCO(args.data.ann_val_file)
label2catid = get_label2catid(coco_gt)

print("Initializing model...")
backbone = DINOv3STAs(**args.backbone)
encoder = HybridEncoder(**args.encoder)
decoder = OVDEIMDecoder(**args.decoder)    
model = OVDEIM(backbone, encoder, decoder, **args.model)

postprocessor = OVDEIMPostProcessor(
    num_classes=args.data.num_classes, 
    num_top_queries=args.decoder.num_queries
)

checkpoint_path = getattr(args, 'pretrained_path', None) or getattr(args, 'checkpoint_path', None)

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 尝试加载 EMA 模型权重（训练时保存的最佳权重）
        if 'ema_state_dict' in checkpoint:
            print("Found EMA state_dict in checkpoint")
            model_state_dict = checkpoint['ema_state_dict']
            # 检查是否是模块化的 EMA 状态
            if 'module' in model_state_dict:
                model_state_dict = model_state_dict['module']
                print("Loading from EMA module state_dict")
            # 跳过 denoising 相关的权重
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if 'decoder.denoising_class_embed.weight' in k:
                    continue
                # 移除 'module.' 前缀（如果存在）
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded EMA model weights")
        # 尝试加载普通模型权重
        elif 'model_state_dict' in checkpoint:
            print("Found model_state_dict in checkpoint")
            model_state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in model_state_dict.items():
                # 移除 'module.' 前缀（如果存在）
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded model weights")
        # 直接加载整个 checkpoint
        else:
            print("Loading checkpoint as state_dict")
            new_state_dict = {}
            for k, v in checkpoint.items():
                # 移除 'module.' 前缀（如果存在）
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded checkpoint weights")
            
        # 打印 checkpoint 信息
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'best_ap' in checkpoint:
            print(f"Best AP in checkpoint: {checkpoint['best_ap']:.4f}")
            
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        print("Continuing with randomly initialized model")
        import traceback
        traceback.print_exc()
else:
    print("Warning: No checkpoint specified or checkpoint not found")
    print("Using randomly initialized model - results will be poor!")

if torch.cuda.is_available():
    device = 'cuda:0'
    print(f"Using GPU: {device}")
else:
    device = 'cpu'
    print("CUDA not available, using CPU")

print("\n" + "="*50)
print("Starting Model Evaluation")
print("="*50 + "\n")
metrics = evaluate(model, postprocessor, test_dataloader, label2catid, coco_gt, device, epoch_or_step="initial")

print("\n" + "="*50)
if metrics:
    print("Evaluation completed successfully!")
    print(f"Final Results: AP={metrics['AP']:.4f}, AP50={metrics['AP50']:.4f}")
else:
    print("Evaluation failed or returned no metrics")
print("="*50)
