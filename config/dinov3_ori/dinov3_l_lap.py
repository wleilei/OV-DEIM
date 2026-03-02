from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class DataConfig:   
    num_classes: int = 1203
    num_training_classes: int = 150
    img_scale: Tuple[int, int] = field(default_factory=lambda: [640, 640])
    prob_aug: float = 1.0
    
    mixup_prob: float = 0.0 # contorl the mixup aug
    apply_moasic: bool = False
    
    blank_text: bool = False # contorl the neg texts
    
    num_workers: int = 4
    
    data_o365_root: str = "/datassd/OG/Objects365_v1/train/"
    ann_o365_file: str = "/datassd/OG/Objects365_v1/objects365_train.json"
    data_gqa_root: str = "/datassd/OG/gqa/images/"
    ann_gqa_file: str = "/datassd/OG/gqa/final_mixed_train_no_coco.json"
    data_flickr_root: str = "/datassd/OG/flickr30k_entities/flickr30k_images/"
    ann_flickr_file: str = "/datassd/OG/flickr30k_entities/final_flickr_separateGT_train.json"
    data_lvis_root: str = "data"
    ann_lvis_file: str = "data/lvis_v1_minival_inserted_image_name.json"

    cache_file_o365: str = "data/o365_text_embeddings.pth"
    cache_file_og: str = "data/og_text_embeddings.pth"
    cache_file_lvis: str = "data/lvis_text_embeddings.pth"

    o365_dir_text: str = "data/o365_text_list.json"
    global_dir_text: str = "data/global_neg_cat.json" # 这里修改负样本来源
    
    class_text_lvis_path: str = "data/lvis_v1_class_texts.json"
    
@dataclass
class BackboneConfig:
    name: str = 'dinov3_vits16'
    weights_path: str = 'weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    interaction_indexes: List[int] = field(default_factory=lambda: [5, 8, 11])   # only need the [1/8, 1/16, 1/32]
    finetune: bool = True
    conv_inplane: int = 32
    hidden_dim: int = 224

@dataclass
class EncoderConfig:
    in_channels: List[int] = field(default_factory=lambda: [224, 224, 224])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])

    # intra
    hidden_dim: int = 224
    use_encoder_idx: List[int] = field(default_factory=lambda: [2])
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 896
    dropout: float = 0.
    enc_act: str = 'gelu'
  
    # cross
    expansion: float = 1.0
    depth_mult: float = 1
    act: str = 'silu'
    
@dataclass
class DecoderConfig:
    feat_channels: List[int] = field(default_factory=lambda: [224, 224, 224])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    num_classes: int = 150
    hidden_dim: int = 224
    dim_feedforward: int = 1792
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
    img_dim: int = 224
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
	    {'params': r'^(?=.*backbone)(?=.*dinov3)(?!.*(?:norm|bn|bias|sem_scale)).*$', 'lr': 1.25e-5},
        {'params': r'^(?=.*backbone)(?=.*dinov3)(?=.*(?:norm|bn|bias|sem_scale)).*$', 'lr': 1.25e-5, 'weight_decay': 0.0},
        {'params': r'^(?=.*backbone)(?!.*dinov3)(?!.*(?:norm|bn|bias|sem_scale)).*$', 'lr': 5e-4},
        {'params': r'^(?=.*backbone)(?!.*dinov3)(?=.*(?:norm|bn|bias|sem_scale)).*$', 'lr': 5e-4, 'weight_decay': 0.0},
        {'params': r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias|lang_bias|lang_scale)).*$', 'weight_decay': 0.0},
    ])
    lr: float = 5e-4    
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 2.5e-4
    eps: float = 1e-12
    eta_min: float = 2.5e-4
    max_norm: float = 0.1

@dataclass
class DINOv3LConfig:
    config_name: str = "base_l"
    
    gpu_ids: List[int] = field(default_factory=lambda: [4,5,6,7])
    nproc_per_node: int = 8
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = '127.0.0.1'
    master_port: str = '11351'
    min_memory_mb: int = 20000
    train_datasets: str = "og" # o365 or og
    epochs: int = 20 # 30
    warmup_epochs: int = 1
    resume: bool = False  # True
    checkpoint_path: str = f"checkpoints/dinov3_{config_name}.pth"
    lighter_aug: int = 20
    batch_size: int = 16
    constant_epochs: int = 5
    cosine_epochs: int = 20
        
    use_swanlab: bool = True
    
    collate_func: str = "train_collate"
    pipeline_type: str = "aug"  # "base" or "aug"

    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    criterion: OVDEIMCriterionConfig = field(default_factory=OVDEIMCriterionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
