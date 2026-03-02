from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class DataConfig:   
    num_classes: int = 80
    num_training_classes: int = 80
    img_scale: Tuple[int, int] = field(default_factory=lambda: [640, 640])
    
    mixup_prob: float = 0.5 # contorl the mixup aug
    mixup_dataset: str = 'coco' # goldg
    apply_moasic: bool = False
        
    blank_text: bool = False # contorl the neg texts
    
    num_workers: int = 4
    
    data_train_root: str = "/datassd/COCO/train2017/"
    ann_train_file: str = "/datassd/COCO/annotations/instances_train2017.json"
    data_val_root: str = "/datassd/COCO/val2017/"
    ann_val_file: str = "/datassd/COCO/annotations/instances_val2017.json"
    
    cache_dir: str = 'data/coco_text_embeddings.pth'
    detect_dir_text: str = "data/coco_texts.json"
    
    
@dataclass
class BackboneConfig:
    name: str = 'vit_tiny'
    weights_path: str = 'weights/dinov3/vitt_distill.pt'
    interaction_indexes: List[int] = field(default_factory=lambda: [3, 7, 11])   # only need the [1/8, 1/16, 1/32]
    conv_inplane: int = 16
    embed_dim: int = 192
    num_heads: int = 3

@dataclass
class EncoderConfig:
    in_channels: List[int] = field(default_factory=lambda: [192, 192, 192])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])

    # intra
    hidden_dim: int = 192
    use_encoder_idx: List[int] = field(default_factory=lambda: [2])
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.
    enc_act: str = 'gelu'
  
    # cross
    expansion: float = 0.34
    depth_mult: float = 0.67
    act: str = 'silu'

    
@dataclass
class DecoderConfig:
    feat_channels: List[int] = field(default_factory=lambda: [192, 192, 192])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    num_classes: int = 150
    hidden_dim: int = 192
    dim_feedforward: int = 512
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0 # 1.0 0.4
    learn_query_content: bool = False
    activation: str = 'silu'
    eval_idx: int = -1
    num_points: List[int] = field(default_factory=lambda: [4, 6, 4]) # [3,3,3] [2,2,2]
    cross_attn_method: str = 'default' # default, discrete
    query_select_method: str = 'default' # default, agnostic 
    num_enc_queries: int = 0
    

@dataclass
class ModelConfig:
    img_dim: int = 192
    text_dim: int = 512

@dataclass
class OVDEIMCriterionConfig:
    weight_dict: Dict = field(default_factory=lambda: {
        'loss_mal': 1,
        'loss_bbox': 5, 
        'loss_giou': 2
    })
    losses: List[str] = field(default_factory=lambda: ['mal', 'boxes'])
    alpha: float = 0.5
    gamma: float = 1.5
    num_classes: int = 150

@dataclass
class OptimizerConfig:
    patterns: List[Dict] = field(default_factory=lambda: [
	    {'params': r'^(?=.*text_adapter)(?!.*(?:norm|bn|bias)).*$', 'lr': 5e-4},
        {'params': r'^(?=.*text_adapter)(?=.*(?:norm|bn|bias)).*$', 'lr': 5e-4, 'weight_decay': 0.0},
	    {'params': r'^(?=.*backbone)(?=.*dinov3)(?!.*(?:norm|bn|bias)).*$', 'lr': 2.5e-5},
        {'params': r'^(?=.*backbone)(?=.*dinov3)(?=.*(?:norm|bn|bias)).*$', 'lr': 2.5e-5, 'weight_decay': 0.0},
        {'params': r'^(?=.*backbone)(?!.*dinov3)(?!.*(?:norm|bn|bias)).*$', 'lr': 5e-4},
        {'params': r'^(?=.*backbone)(?!.*dinov3)(?=.*(?:norm|bn|bias)).*$', 'lr': 5e-4, 'weight_decay': 0.0},
        {'params': r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias|lang_bias|lang_scale)).*$', 'weight_decay': 0.0},
    ])
    lr: float = 5e-4    
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 2.5e-4
    eps: float = 1e-12
    eta_min: float = 2.5e-4
    max_norm: float = 0.1

@dataclass
class DINOv3SConfig:
    config_name: str = "fintune_coco"
    
    gpu_ids: List[int] = field(default_factory=lambda: [4,5,6,7])
    nproc_per_node: int = 2
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = '127.0.0.1'
    master_port: str = '11352'
    min_memory_mb: int = 22000
    train_datasets: str = "og" # o365 or og
    epochs: int = 24 # 30
    warmup_epochs: int = 1
    resume: bool = False  # True
    checkpoint_path: str = f"checkpoints/dinov3l_{config_name}.pth"
    
    batch_size: int = 24
    lighter_aug: int = 24
    constant_epochs: int = 5
    copy_paste_epoch: int = 24 # close the copy_paste aug
    
    collate_func: str = "train_collate"
    
    use_swanlab: bool = True
    
    pretrained: bool = True
    pretrained_path: str = 'pretrained_ckp/dinov3_base_l_train_collate_cp_16_64_grid_synergism_3385.pth'

    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    criterion: OVDEIMCriterionConfig = field(default_factory=OVDEIMCriterionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
