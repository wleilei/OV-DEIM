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


def train(rank, world_size, args, gpu_ids):
    # Set up distributed training environment
    setup(rank, world_size, args, gpu_ids)
    if not args.resume:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = args.checkpoint_path.replace('.pth', f'_{args.collate_func}_{current_time}.pth')
    else:
        checkpoint_path = args.checkpoint_path

    ##### data loading #####
    train_pipeline_aug, train_pipeline_base, test_pipeline = get_pipelines(size=args.data.img_scale, 
                                                                    num_texts=args.data.num_training_classes, 
                                                                    blank_text=args.data.blank_text,
                                                                    mixup_prob=args.data.mixup_prob,
                                                                    apply_moasic=args.data.apply_moasic,
                                                                    )
    if args.pipeline_type == "aug":
        print('using aug pipeline')
        train_pipeline = train_pipeline_aug
    else:
        print('using base pipeline')
        train_pipeline = train_pipeline_base

    if args.train_datasets == "o365":
        train_dataset = get_o365_ori_dataset(data_o365_root=args.data.data_o365_root, 
                                             ann_o365_file=args.data.ann_o365_file, 
                                    pipeline_aug=train_pipeline_aug,
                                    pipeline_clean=train_pipeline_base,
                                    cache_dir=args.data.cache_file_o365,
                                    o365_dir_text=args.data.o365_dir_text
                                        )
    elif args.train_datasets == "og":
        train_dataset = get_og_ori_dataset(data_o365_root=args.data.data_o365_root, 
                                           ann_o365_file=args.data.ann_o365_file, 
                                data_gqa_root=args.data.data_gqa_root, ann_gqa_file=args.data.ann_gqa_file, 
                                data_flickr_root=args.data.data_flickr_root, ann_flickr_file=args.data.ann_flickr_file,
                                cache_dir=args.data.cache_file_og,
                                pipeline_aug=train_pipeline,
                                pipeline_clean=train_pipeline_base,
                                o365_dir_text=args.data.o365_dir_text,
                                global_dir_text=args.data.global_dir_text
                                )
        
    test_dataset = get_lvis_ori_dataset(data_root=args.data.data_lvis_root, ann_file=args.data.ann_lvis_file, 
                                    pipeline_clean=test_pipeline, 
                                    cache_dir=args.data.cache_file_lvis,
                                    lvis_path=args.data.class_text_lvis_path
                                )

    module = importlib.import_module("dataloader.data_utils")
    func_name = args.collate_func
    assert hasattr(module, func_name), f"Unknown collate fn: {func_name}"
    train_collate_synthetic = getattr(module, func_name)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  prefetch_factor=4,                       
                                  collate_fn=train_collate, sampler=train_sampler, 
                                  num_workers=args.data.num_workers)
    synthetic_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    synthetic_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  prefetch_factor=4,                       
                                  collate_fn=train_collate_synthetic, sampler=synthetic_sampler, 
                                  num_workers=args.data.num_workers)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8,
                                 collate_fn=eval_collate, sampler=test_sampler, num_workers=2)

    lvis_gt = LVIS(args.data.ann_lvis_file)

    ##### model #####
    backbone = DINOv3STAs(**args.backbone)
    encoder = HybridEncoder(**args.encoder)
    decoder = OVDEIMDecoder(**args.decoder)    
    model = OVDEIM(backbone, encoder, decoder, **args.model).to(gpu_ids[rank])
    if args.nproc_per_node * args.nnodes > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    postprocessor = OVDEIMPostProcessor(num_classes=args.data.num_classes, 
                                        num_top_queries=args.decoder.num_queries)
    model = DDP(model, device_ids=[gpu_ids[rank]],find_unused_parameters=False)

    ##### loss function and optimizer #####
    criterion = OVDEIMCriterion(**args.criterion)
    param_groups = get_optim_params(args.optimizer.patterns, model.named_parameters())
    optimizer = optim.AdamW(param_groups, lr=args.optimizer.lr, 
                            betas=args.optimizer.betas, weight_decay=args.optimizer.weight_decay)

        # Training configuration
    epochs = args.epochs
    steps_per_epoch = len(train_dataloader)
    total_steps = args.cosine_epochs * steps_per_epoch
    warmup_epochs = getattr(args, 'warmup_epochs', 1)
    warmup_steps = warmup_epochs * steps_per_epoch
    constant_epochs = args.constant_epochs - warmup_epochs
    constant_steps = constant_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps - constant_steps
    warmup_scheduler = WarmupConstantCosine(optimizer, warmup_steps=warmup_steps, 
                                            constant_steps=constant_steps, 
                                            cosine_steps=cosine_steps, 
                                            max_lr=args.optimizer.lr, 
                                            min_lr=args.optimizer.eta_min)
    ema = ModelEMA(model, decay=0.9999, warmups=warmup_steps)
    scaler = GradScaler()
    
    # --- Checkpoint Loading Logic ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        if rank == 0:
            print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        # Map location to current device to avoid loading issues
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_ids[rank]}
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Adjust model state dict keys if saved with DDP wrapper
        model_state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.module.load_state_dict(new_state_dict) # Load into the underlying model

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'ema_state_dict' in checkpoint and ema:
            ema_state_dict = checkpoint['ema_state_dict']
            try:
                # Load directly into the EMA object itself
                ema.load_state_dict(ema_state_dict)
                if rank == 0:
                    print("Successfully loaded EMA state.")
            except RuntimeError as e:
                # Log potential issues if direct loading fails
                if rank == 0:
                    print(f"Rank {rank}: Warning - Error loading EMA state_dict directly: {e}. Check EMA implementation details if issues persist.")
                    print(f"Rank {rank}: EMA state might not be fully restored.")
            # --- End Corrected EMA Loading ---
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"Rank {rank}: Fast-forwarding scheduler/warmup to step {global_step}")
        global_step = start_epoch * steps_per_epoch # Estimate step based on resumed epoch
        for _ in range(global_step):
            warmup_scheduler.step() # Call step() the required number of times
        if rank == 0:
            print(f"Resumed from epoch {start_epoch-1}. Starting epoch {start_epoch}. Global step approx {global_step}.")
            print(f"Current LR after resuming and fast-forwarding: {optimizer.param_groups[0]['lr']}")

    # Initialize SwanLab on rank 0
    use_swanlab = args.use_swanlab
    if rank == 0 and use_swanlab:
        import swanlab
        swanlab_instance = swanlab.init(
            project="rt-ovdt-grid",
            experiment_name=f"{args.config_name}_{args.collate_func}_{int(time.time())}",
            workspace="ovod"
        )
        
        total_params = count_parameters(model.module, swanlab_instance)
        print(f"Model initialized with {total_params:,} parameters")
        
        log_param_groups_to_swanlab(param_groups, model.named_parameters(), swanlab_instance)
    else:
        swanlab_instance = None
    
    best_ap = 0.0
    best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
    if rank == 0:
        print(f"Best checkpoint will be saved to: {best_checkpoint_path}")
        
        if args.resume:
            try:
                best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
                if 'best_ap' in best_checkpoint:
                    best_ap = best_checkpoint['best_ap']
                    print(f"Resumed with best AP: {best_ap:.4f}")
            except FileNotFoundError:
                print(f"No existing best checkpoint found at {best_checkpoint_path}, starting fresh")
            except Exception as e:
                print(f"Error loading best checkpoint info: {e}")

    model.train()
    num_iters = len(train_dataloader)
    use_synthetic = True
    for epoch in range(start_epoch, epochs):
        train_dataset.epoch = epoch
        if epoch == args.lighter_aug:
            if rank == 0:
                print(f"Switching to train_pipeline_base at epoch {epoch+1}")
            for i in range(len(train_dataset.datasets)):
                train_dataset.datasets[i].pipeline_aug = train_pipeline_base
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_collate, sampler=train_sampler, num_workers=6)
            dist.barrier()  # Synchronize all processes to ensure consistent pipeline switch
            num_iters = len(train_dataloader)
            use_synthetic = False
        train_sampler.set_epoch(epoch)
        synthetic_sampler.set_epoch(epoch)
        train_data_iter = iter(train_dataloader)
        synthetic_data_iter = iter(synthetic_dataloader)
        for i in tqdm(range(num_iters), desc=f"Epoch {epoch+1}/{epochs}"):
            if random.uniform(0, 1.0) < 0.75 or not use_synthetic:
                batch = next(train_data_iter)
            else:
                batch = next(synthetic_data_iter)
            data = batch['inputs'].to(gpu_ids[rank]).to(torch.float32)
            targets = {k: v.to(gpu_ids[rank]) if isinstance(v, torch.Tensor) else v for k, v in batch['data_samples'].items()}
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(data, targets)
                loss_dict = criterion(outputs, targets)    
                loss_dec = sum(loss_dict[k] for k in ['loss_mal', 'loss_bbox', 'loss_giou'])
                denoising_keys = [k for k in loss_dict.keys() if '_dn_' in k]
                if denoising_keys:
                    loss_denoising = sum(loss_dict[k].sum() for k in denoising_keys)
                else:
                    loss_denoising = torch.tensor(0.0, device=gpu_ids[rank])
                loss_aux = sum(loss_dict[k].sum() for k in loss_dict.keys() if k not in ['loss_mal', 'loss_bbox', 'loss_giou'] and '_dn_' not in k)
                loss = loss_dec + loss_denoising + loss_aux

            skip_step = torch.tensor(0.0, device=gpu_ids[rank])
            if torch.isnan(loss) or torch.isinf(loss):
                skip_step = torch.tensor(1.0, device=gpu_ids[rank])
                print(f"Warning: Invalid loss detected at epoch {epoch}, step {i} on rank {rank}")
            dist.all_reduce(skip_step, op=dist.ReduceOp.MAX)

            if skip_step.item() > 0:
                if rank == 0:
                    print(f"Skipping step due to invalid loss on one or more GPUs.")
                global_step +=1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.optimizer.max_norm)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            warmup_scheduler.step()
            
            if global_step % 100 == 0:
                lr_dict = {f"learning_rate_group_{i}": group['lr'] for i, group in enumerate(optimizer.param_groups)}
                
                metrics_tensors = {
                    "total_loss": loss,
                    "loss_dec": loss_dec,
                    "loss_denoising": loss_denoising,
                }
                
                for k, v in loss_dict.items():
                    metrics_tensors[f"loss_dict_{k}"] = v
                
                for k, v in metrics_tensors.items():
                    dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    metrics_tensors[k] = v / world_size
                
                metrics = {
                    "total_loss": metrics_tensors["total_loss"].item(),
                    "loss_dec": metrics_tensors["loss_dec"].item(),
                    "loss_denoising": metrics_tensors["loss_denoising"].item(),
                }
                
                for k, v in loss_dict.items():
                    metrics[k] = metrics_tensors[f"loss_dict_{k}"].item()
                
                metrics.update(lr_dict)
                if rank == 0 and use_swanlab and swanlab_instance:
                    swanlab_instance.log(metrics)
            global_step += 1
        # Evaluation at the end of each epoch
        if epoch % 1 == 0:
            if rank == 0:
                state = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict()
                }
                torch.save(state, checkpoint_path)
                # if (epoch+1) % 1 == 0:
                #     torch.save(state, checkpoint_path.replace('.pth', f'_{epoch+1}.pth'))
        torch.cuda.empty_cache()
        module = ema.module if ema else model
        module.eval()
        if rank == 0:
            print(f"Evaluating at Epoch {epoch+1}")
        predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                data = batch['inputs'].to(gpu_ids[rank]).to(torch.float32)
                targets = {k: v.to(gpu_ids[rank]) if isinstance(v, torch.Tensor) else v for k, v in batch['data_samples'].items()}
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

        # Gather predictions across all processes
        all_predictions = [None] * world_size
        dist.all_gather_object(all_predictions, predictions)
        if rank == 0:
            all_predictions = [item for sublist in all_predictions for item in sublist]
            lvis_eval = LVISEval(lvis_gt, all_predictions, iou_type='bbox')
            lvis_eval.run()
            eval_results = lvis_eval.get_results()
            lvis_eval.print_results()
            current_ap = eval_results['AP']
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
            
            if current_ap > best_ap:
                best_ap = current_ap
                print(f"New best AP: {best_ap:.4f} at epoch {epoch+1}")
                print(f"Saving best checkpoint to: {best_checkpoint_path}")
                
                best_state = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'best_ap': best_ap,
                    'eval_results': eval_results
                }
                torch.save(best_state, best_checkpoint_path)
                print(f"Best checkpoint saved successfully!")
            else:
                print(f"Current AP: {current_ap:.4f}, Best AP: {best_ap:.4f}")
            
            if use_swanlab and swanlab_instance:
                swanlab_instance.log(metrics, step=epoch)
                swanlab_instance.log({"best_ap": best_ap}, step=epoch)

        dist.barrier()
        model.train()

    # Cleanup
    if rank == 0 and use_swanlab and swanlab_instance:
        swanlab_instance.finish()
    dist.destroy_process_group()


def main(args: DictConfig):
    try:
        while True:
            available_gpus = get_available_gpus(min_memory_mb=args.min_memory_mb)
            if len(available_gpus) < args.nproc_per_node:
                print(f"Waiting for {args.nproc_per_node} GPUs, {len(available_gpus)} available...")
                time.sleep(10)
            else:
                break

        gpu_ids = available_gpus[:args.nproc_per_node]
        # gpu_ids = args.gpu_ids
        world_size = args.nproc_per_node * args.nnodes
        mp.spawn(train, args=(world_size, args, gpu_ids), nprocs=args.nproc_per_node, join=True,)

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_l")
    parser.add_argument("--collate_func", type=str, default="train_collate")
    parser.add_argument("--pipeline_type", type=str, default="aug", choices=["aug", "base"], help="Pipeline type: 'aug' or 'base'")
    args = parser.parse_args()
    config = hydra.compose(config_name=args.config)
    config.collate_func = args.collate_func
    config.pipeline_type = args.pipeline_type
    main(config)
