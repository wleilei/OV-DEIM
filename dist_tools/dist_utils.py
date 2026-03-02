import pynvml
import torch
import os
import torch.distributed as dist
import psutil
import random
import numpy as np
from datetime import timedelta

def get_available_gpus(min_memory_mb=1000):
    # Check CPU RAM (convert bytes to GB)
    total_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    if total_ram_gb < 150:
        print(f"Insufficient available CPU RAM: {total_ram_gb:.2f}GB available, 150GB required")
        return []

    # Proceed with GPU checking if RAM requirement is met
    if not torch.cuda.is_available():
        return []
    
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    available_gpus = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        free_memory_mb = mem_info.free / 1024 / 1024
        gpu_util = utilization.gpu
        
        if free_memory_mb >= min_memory_mb and gpu_util < 150:
            available_gpus.append(i)
    
    pynvml.nvmlShutdown()
    return available_gpus


def setup(rank, world_size, args, gpu_ids, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    
    # Ensure deterministic behavior (might impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check and create checkpoints directory if it doesn't exist
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        if rank == 0:  # Only print from rank 0 to avoid duplicate messages
            print(f"Created directory: {checkpoints_dir}")

    # Original setup code
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    global_rank = args.node_rank * args.nproc_per_node + rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=global_rank)
    torch.cuda.set_device(gpu_ids[rank])

def init_dist_from_env(seed=42, create_checkpoint_dir=True):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if create_checkpoint_dir:
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)
    
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[DDP] Already initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank

    torchrun_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if torchrun_env:
        return _init_torchrun_dist()
    else:
        print("[DDP] No distributed environment detected, running in single-process mode.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  
        return 0, 1, 0

def _init_torchrun_dist():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    print(f"[torchrun] Initializing: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30)
        )
    except Exception as e:
        print(f"[torchrun] Failed to initialize process group: {e}")
        raise

    if rank == 0:
        print(f"[torchrun] Successfully initialized distributed training")

    return rank, world_size, local_rank
