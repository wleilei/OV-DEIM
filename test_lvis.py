import logging
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import hydra 
from omegaconf import DictConfig

from config.dinov3_ori.dinov3_l import DINOv3LConfig as base_l
from config.dinov3_ori.dinov3_m import DINOv3MConfig as base_m
from config.dinov3_ori.dinov3_s import DINOv3SConfig as base_s

from hydra.core.config_store import ConfigStore
import time
import datetime
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from lvis import LVIS, LVISEval
import torch.nn as nn
import importlib
import random

from dataloader.data_utils import (get_pipelines, get_o365_ori_dataset, 
                              get_og_ori_dataset, get_lvis_ori_dataset, train_collate, eval_collate)
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
cs.store(name="base_l", node=base_l)
cs.store(name="base_m", node=base_m)
cs.store(name="base_s", node=base_s)
hydra.initialize(config_path="config", version_base=None)

args = hydra.compose(config_name="base_l")

_, _, test_pipeline = get_pipelines(size=args.data.img_scale, 
                                    num_texts=args.data.num_training_classes, 
                                    blank_text=args.data.blank_text,
                                    mixup_prob=args.data.mixup_prob,
                                    apply_moasic=args.data.apply_moasic,
                                    )

test_dataset = get_lvis_ori_dataset(data_root=args.data.data_lvis_root, ann_file=args.data.ann_lvis_file, 
                                    pipeline_clean=test_pipeline, 
                                    cache_dir=args.data.cache_file_lvis,
                                    lvis_path=args.data.class_text_lvis_path
                                )

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=eval_collate, num_workers=2)

lvis_gt = LVIS(args.data.ann_lvis_file)

device = "cuda:4"

backbone = DINOv3STAs(**args.backbone)
encoder = HybridEncoder(**args.encoder)
decoder = OVDEIMDecoder(**args.decoder)    
model = OVDEIM(backbone, encoder, decoder, **args.model).to(device)

postprocessor = OVDEIMPostProcessor(num_classes=args.data.num_classes, 
                                    num_top_queries=args.decoder.num_queries)

checkpoint_path = "weights/ovdeim_l.pth"

model_state_dict = torch.load(checkpoint_path, map_location=device)

new_state_dict = {}
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v # remove 'module.' prefix
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict) # Load into the underlying model
model.to(device)

from collections import defaultdict
from lvis import LVIS, LVISResults, LVISEval

def eval_fixed_ap(num_enc_queries):
    decoder.num_enc_queries = num_enc_queries
    module = model
    module.eval()
    predictions = []
    postprocessor = OVDEIMPostProcessor(num_classes=args.data.num_classes, 
                                        num_top_queries=300+num_enc_queries)
    with torch.no_grad():
        for batch in test_dataloader:
            data = batch['inputs'].to(device).to(torch.float32)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['data_samples'].items()}
            outputs, enc_outputs = module(data, targets)
            outputs['pred_logits'] = torch.cat([outputs['pred_logits'], enc_outputs[0]], dim=1)
            outputs['pred_boxes'] = torch.cat([outputs['pred_boxes'], enc_outputs[1]], dim=1)
            preds = postprocessor(outputs, targets)
            for pred, img_id in zip(preds, targets['img_id']):
                for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                    predictions.append({
                        'image_id': img_id,
                        'category_id': label.item()+1,
                        'bbox': box.tolist(),
                        'score': score.item()
                    })
    topk = 10000
    by_cat = defaultdict(list)
    for ann in predictions:
        by_cat[ann["category_id"]].append(ann)
    results = []
    missing_dets_cats = set()
    for cat, cat_anns in by_cat.items():
        if len(cat_anns) < topk:
            missing_dets_cats.add(cat)
        results.extend(sorted(cat_anns, key=lambda x: x["score"], reverse=True)[:topk])
    results = LVISResults(lvis_gt, results, max_dets=300+num_enc_queries)
    lvis_eval = LVISEval(lvis_gt, results, iou_type='bbox',)
    params = lvis_eval.params
    params.max_dets = 300+num_enc_queries   # No limit on detections per image.
    lvis_eval.run()
    eval_results = lvis_eval.get_results()
    # lvis_eval.print_results()
    metrics = {
        "AP": eval_results['AP'],
        "AP50": eval_results['AP50'],
        "AP75": eval_results['AP75'],
        "APs": eval_results['APs'],
        "APm": eval_results['APm'],
        "APl": eval_results['APl'],
        "APr": eval_results['APr'],
        "APc": eval_results['APc'],
        "APf": eval_results['APf'],
    }
    print(metrics)
    
eval_fixed_ap(num_enc_queries=700)

def eval_ap():
    decoder.num_enc_queries = 0
    module = model
    module.eval()
    predictions = []
    postprocessor = OVDEIMPostProcessor(num_classes=args.data.num_classes, 
                                        num_top_queries=300)
    with torch.no_grad():
        for batch in test_dataloader:
            data = batch['inputs'].to(device).to(torch.float32)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['data_samples'].items()}
            outputs = module(data, targets)
            preds = postprocessor(outputs, targets)
            for pred, img_id in zip(preds, targets['img_id']):
                for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                    predictions.append({
                        'image_id': img_id,
                        'category_id': label.item()+1,
                        'bbox': box.tolist(),
                        'score': score.item()
                    })
    lvis_eval = LVISEval(lvis_gt, predictions, iou_type='bbox')
    lvis_eval.run()
    eval_results = lvis_eval.get_results()
    lvis_eval.print_results()
    metrics = {
        "AP": eval_results['AP'],
        "AP50": eval_results['AP50'],
        "AP75": eval_results['AP75'],
        "APs": eval_results['APs'],
        "APm": eval_results['APm'],
        "APl": eval_results['APl'],
        "APr": eval_results['APr'],
        "APc": eval_results['APc'],
        "APf": eval_results['APf'],
    }
    print(metrics)   
    
eval_ap()
