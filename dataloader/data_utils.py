from typing import Dict, Tuple, Any
import torch
import torchvision
import numpy as np

torchvision.disable_beta_transforms_warning()

from .dataset import ConcatDataset, MultiModalDataset, CacheMultiModalDataset
from .dataset import CocoDataset
from .transforms import (Compose, KeepRatioResize, LetterResize, 
                         RandomAffine, Augmentation, 
                         MultiModalMoasic, RandomLoadTexts, ToTensor, 
                         SanitizeBoundingBoxes,LimitBboxes, Mixup, BackImg, RandomHorizontalFlip)
from tqdm import tqdm
import random
from .transforms import ObjectResize, RandomIoUCrop
import copy
from functools import partial

random_horizontal_flip = RandomHorizontalFlip(p=0.5)
random_iou_crop = RandomIoUCrop()
object_resize_11 = ObjectResize((640, 640), pad_val = dict(img=0, mask=0, seg=255))
object_resize_12 = ObjectResize((640, 320), pad_val = dict(img=0, mask=0, seg=255))
object_resize_21 = ObjectResize((320, 640), pad_val = dict(img=0, mask=0, seg=255))
object_resize_22 = ObjectResize((320, 320), pad_val = dict(img=0, mask=0, seg=255))

object_resize_44 = ObjectResize((160, 160), pad_val = dict(img=0, mask=0, seg=255))
object_resize_48 = ObjectResize((160, 80), pad_val = dict(img=0, mask=0, seg=255))
object_resize_84 = ObjectResize((80, 160), pad_val = dict(img=0, mask=0, seg=255))
object_resize_88 = ObjectResize((80, 80), pad_val = dict(img=0, mask=0, seg=255))

def get_pipelines(size=(640, 640), num_texts=80, blank_text=False,
                  mixup_prob=0.5, apply_moasic = False
                  ):
    """Get training and testing pipelines."""
    train_pipeline_aug = Compose([
        MultiModalMoasic(size=size, center_ratio_range=(0.5, 1.5), min_bbox_size=1.0),
        RandomAffine(max_rotate_degree=0.0, max_translate_ratio=0.1, 
                     scaling_ratio_range=(0.5, 1.5), max_shear_degree=0.0, max_aspect_ratio=100.0,
                     border=(-320, -320), border_val=(114, 114, 114), bbox_clip_border=True),
        Mixup(prob=mixup_prob, size=size, apply_moasic=apply_moasic),
        Augmentation(),
        SanitizeBoundingBoxes(min_size=1.0),
        RandomLoadTexts(num_texts=num_texts, blank_text=blank_text),
        LimitBboxes(max_bboxes=300),
        # BackImg(size=size),
        ToTensor()
    ])

    train_pipeline_base = Compose([
        KeepRatioResize(size=size),
        LetterResize(size=size),
        RandomAffine(max_rotate_degree=0.0, max_shear_degree=0.0,
                    scaling_ratio_range=(0.5, 1.5), max_aspect_ratio=100.0,
                    border_val=(114, 114, 114)),
        Augmentation(),
        SanitizeBoundingBoxes(min_size=1.0),
        RandomLoadTexts(num_texts=num_texts, blank_text=blank_text),
        LimitBboxes(max_bboxes=300),
        ToTensor()
    ])

    test_pipeline = Compose([
        KeepRatioResize(size=size),
        LetterResize(size=size),
        ToTensor()
    ])

    return train_pipeline_aug, train_pipeline_base, test_pipeline

def get_coco_pipelines(size=(640, 640), num_texts=80, 
                       blank_text=False, mixup_prob=0.5,
                       apply_moasic: bool = False
                       ):
    """Get training and testing pipelines."""
    train_pipeline_1 = Compose([
        MultiModalMoasic(size=size, center_ratio_range=(0.5, 1.5), min_bbox_size=1.0,),
        RandomAffine(max_rotate_degree=0.0, max_translate_ratio=0.1, 
                     scaling_ratio_range=(0.5, 1.5), max_shear_degree=0.0, 
                     border=(-320, -320), border_val=(114, 114, 114), bbox_clip_border=True),
        Augmentation(),
        Mixup(prob=mixup_prob, size=size,
              apply_moasic=apply_moasic),
        SanitizeBoundingBoxes(min_size=1.0),
        RandomLoadTexts(num_texts=num_texts, blank_text=blank_text),
        LimitBboxes(max_bboxes=300),
        ToTensor()
    ])
    train_pipeline_2 = Compose([
        KeepRatioResize(size=size),
        LetterResize(size=size),
        RandomAffine(max_rotate_degree=0.0, max_shear_degree=0.0,
                    scaling_ratio_range=(0.5, 1.5), max_aspect_ratio=100.0,
                    border_val=(114, 114, 114)),
        Augmentation(),
        SanitizeBoundingBoxes(min_size=1.0),
        RandomLoadTexts(num_texts=num_texts, blank_text=blank_text),
        LimitBboxes(max_bboxes=300),
        ToTensor()
    ])
    
    train_pipeline_syn = Compose([
        MultiModalMoasic(size=size, center_ratio_range=(0.5, 1.5), min_bbox_size=1.0,),
        RandomAffine(max_rotate_degree=0.0, max_translate_ratio=0.1, 
                     scaling_ratio_range=(0.5, 1.5), max_shear_degree=0.0, 
                     border=(-320, -320), border_val=(114, 114, 114), bbox_clip_border=True),
        Augmentation(),
        SanitizeBoundingBoxes(min_size=1.0),
        RandomLoadTexts(num_texts=num_texts, blank_text=blank_text),
        LimitBboxes(max_bboxes=300),
        ToTensor()
    ])

    test_pipeline = Compose([
        KeepRatioResize(size=size),
        LetterResize(size=size),
        ToTensor()
    ])

    return train_pipeline_1, train_pipeline_2, train_pipeline_syn, test_pipeline

    
def get_o365_ori_dataset(data_o365_root, ann_o365_file, 
                         pipeline_aug, pipeline_clean, cache_dir=None, 
                         o365_dir_text=None, prob_aug = 1.0):
    o365_dataset = MultiModalDataset(data_o365_root, ann_o365_file, 
                                     detection=True, return_img=False)
    o365_dataset = CacheMultiModalDataset(o365_dataset)
    return ConcatDataset([o365_dataset], 
                         pipeline_aug=pipeline_aug, pipeline_clean=pipeline_clean, cache_dir=cache_dir, 
                         detect_dir_text=o365_dir_text, prob_aug=prob_aug)
    
    
def get_og_ori_dataset(data_o365_root, ann_o365_file, 
                   data_gqa_root, ann_gqa_file, 
                   data_flickr_root, ann_flickr_file,
                   pipeline_aug, pipeline_clean, cache_dir=None, 
                   o365_dir_text=None, global_dir_text=None, prob_aug = 1.0
                   ):
    o365_dataset = MultiModalDataset(data_o365_root, ann_o365_file, 
                                     detection=True, return_img=False)
    o365_dataset = CacheMultiModalDataset(o365_dataset)
    gqa_dataset = MultiModalDataset(root=data_gqa_root, annFile=ann_gqa_file, 
                                    detection=False, return_img=False)
    gqa_dataset = CacheMultiModalDataset(gqa_dataset)
    flickr_dataset = MultiModalDataset(root=data_flickr_root, annFile=ann_flickr_file, 
                                       detection=False, return_img=False)
    flickr_dataset = CacheMultiModalDataset(flickr_dataset)
    
    return ConcatDataset([o365_dataset, gqa_dataset, flickr_dataset], 
                         pipeline_aug=pipeline_aug, pipeline_clean=pipeline_clean, cache_dir=cache_dir, 
                         detect_dir_text=o365_dir_text, global_dir_text=global_dir_text, prob_aug = prob_aug
                         )

def get_lvis_ori_dataset(data_root, ann_file, pipeline_clean, cache_dir=None, lvis_path=None):
    lvis_dataset = MultiModalDataset(root=data_root, annFile=ann_file, detection=True, 
                                     lvis=True, lvis_path=lvis_path, return_img=False)
    lvis_dataset = CacheMultiModalDataset(lvis_dataset)
    return ConcatDataset([lvis_dataset], pipeline_clean=pipeline_clean, cache_dir=cache_dir, test=True)

def get_coco_train_dataset(data_train_root, ann_train_file,
                           cache_dir='data/coco_text_embeddings.pth',
                           detect_dir_text="data/coco_texts.json",
                           pipeline_aug=None,
                           pipeline_clean=None,
                           ):
    train_dataset = CocoDataset(data_train_root, ann_train_file)
    train_dataset = CacheMultiModalDataset(train_dataset)
    train_dataset = ConcatDataset([train_dataset], 
                              cache_dir=cache_dir,
                              detect_dir_text=detect_dir_text,
                              pipeline_aug=pipeline_aug,
                              pipeline_clean=pipeline_clean,
                              )
    return train_dataset

def get_coco_test_dataset(data_test_root, ann_test_file,
                          cache_dir = 'data/coco_text_embeddings.pth',
                          pipeline_clean=None):
    test_dataset = CocoDataset(data_test_root, ann_test_file)
    test_dataset = CacheMultiModalDataset(test_dataset)
    test_dataset = ConcatDataset([test_dataset], 
                              cache_dir = cache_dir,
                              pipeline_clean=pipeline_clean,
                              test=True)
    return test_dataset

def get_label2catid(coco):
    cats = coco.loadCats(coco.getCatIds())
    cats_sorted = sorted(cats, key=lambda x: x['id'])

    label2catid= {}

    for i, cat in enumerate(cats_sorted):
        catid = cat['id']
        label2catid[i] = catid
    return label2catid

def train_collate(data_batch: Tuple[np.ndarray, Dict[str, Any]]) -> Dict:
    # Pre-allocate lists for batch data
    batch_imgs = []
    batch_embeddings = []
    batch_bboxes_labels = []
    num_boxes_per_sample = []
    text_ids = []
    # Process each sample in the batch
    for i, sample in enumerate(data_batch):
        inputs = sample[0]
        data_instances = sample[1]['instances']
        embeddings = sample[1]['text_feats']
        
        batch_imgs.append(inputs)
        batch_embeddings.append(embeddings)
        gt_bboxes = data_instances['bboxes']
        gt_labels = data_instances['labels']
        batch_idx = torch.full((len(gt_labels), 1), i, device=gt_labels.device)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes), dim=1)
        batch_bboxes_labels.append(bboxes_labels)
        num_boxes_per_sample.append(len(gt_bboxes))
        text_ids.extend([sample[1]['text_ids'][j] for j in gt_labels.to(torch.int64)])

    # Construct collated results
    collated_results = {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': {
            'text_feats': torch.stack(batch_embeddings, 0),
            'idx': torch.cat(batch_bboxes_labels, 0)[:, 0].to(torch.int64),  # batch indices
            'labels': torch.cat(batch_bboxes_labels, 0)[:, 1].to(torch.int64),  # labels
            'num_boxes': num_boxes_per_sample,  # Number of boxes per sample
            'text_ids': text_ids
        }
    }

    # Process bounding boxes with optimized conversion
    boxes = torch.cat(batch_bboxes_labels, 0)[:, 2:]  # Extract bounding box coordinates (from second column)
    width, height = collated_results['inputs'].shape[2:][::-1]  # [width, height]
    norm_factor = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
    boxes = boxes / norm_factor
    boxes = torch.clamp(boxes, min=0.0, max=1.0)
    collated_results['data_samples']['boxes'] = boxes
    collated_results['data_samples']['norm_factor'] = norm_factor
    
    return collated_results

def get_objects_from_batch(data_batch):
    objects = []
    objects_L = []
    expand_ratio = 0.2
    for i, data in enumerate(data_batch):
        img, targets = data
        labels = targets['instances']['labels']
        for j, box in enumerate(targets['instances']['bboxes']):
            box = box.to(torch.int64)
            x1, y1, x2, y2 = box
            box_w, box_h = x2 - x1, y2 - y1
            expand_w = int(box_w * expand_ratio)
            expand_h = int(box_h * expand_ratio)
            x1_exp = max(x1 - expand_w, 0)
            y1_exp = max(y1 - expand_h, 0)
            x2_exp = min(x2 + expand_w, targets['width'])
            y2_exp = min(y2 + expand_h, targets['height'])

            # relative box position within expanded patch
            rel_x1 = x1 - x1_exp
            rel_y1 = y1 - y1_exp
            rel_x2 = x2 - x1_exp
            rel_y2 = y2 - y1_exp
            
            # crop expanded image patch
            img_crop_box = img[:, y1_exp:y2_exp, x1_exp:x2_exp].clone()
            relative_coord = torch.tensor([rel_x1, rel_y1, rel_x2, rel_y2], dtype=torch.float32)
            text_id = copy.deepcopy(targets['text_ids'][labels[j].to(torch.int64)])
            objects.append((img_crop_box, relative_coord, text_id))
            if box_w >= 160 or box_h >= 160:
                objects_L.append((img_crop_box, relative_coord, text_id))
    return objects, objects_L

def apply_copy_paste(sample, objects, dataset, n_objects=3):
    img, targets = sample
    selected_indices = random.choices(range(len(objects)), k = n_objects)
    img_h, img_w = img.shape[1], img.shape[2]
    boxes = targets['instances']['bboxes'].clone()
    text_ids_o = targets['text_ids'].copy()
    text_ids = [text_ids_o[i] for i in targets['instances']['labels'].to(torch.int64)]
    for idx in selected_indices:
        box_img, box_relative_coord, box_text_id = copy.deepcopy(objects[idx])
        new_w_px, new_h_px = box_img.shape[2], box_img.shape[1]
        x1 = random.randint(0, img_w - new_w_px) if new_w_px < img_w else 0
        y1 = random.randint(0, img_h - new_h_px) if new_h_px < img_h else 0
        copy_img = img[:,y1:y1+new_h_px, x1:x1+new_w_px]
        beta = 0.5          
        img[:,y1:y1+new_h_px, x1:x1+new_w_px] = copy_img * beta + box_img * (1 - beta)

        abs_x1_obj, abs_y1_obj, abs_x2_obj, abs_y2_obj = [box_relative_coord[coord_idx].item() + (x1 if coord_idx in [0,2] else y1) for coord_idx in range(4)]
        new_boxes = torch.tensor([[abs_x1_obj, abs_y1_obj, abs_x2_obj, abs_y2_obj]], dtype=torch.float32)
        boxes = torch.cat([boxes, new_boxes], dim=0)
        text_ids.append(box_text_id)
    text_ids_unique = list(set(text_ids))
    text2idx = {text_id_val: local_idx for local_idx, text_id_val in enumerate(text_ids_unique)}
    num_embedding_slots = targets['text_feats'].shape[0]
    placeholder_embedding = dataset.embeddings[dataset.text_to_index[" "]]
    text_embeddings = placeholder_embedding.repeat(num_embedding_slots, 1)
    num_unique_to_fill = min(len(text_ids_unique), num_embedding_slots)
    if num_unique_to_fill > 0:
        embeddings_to_assign = dataset.embeddings[text_ids_unique[:num_unique_to_fill]]
        text_embeddings[:num_unique_to_fill,:] = embeddings_to_assign
        num_neg_texts = num_embedding_slots - num_unique_to_fill
        if num_neg_texts > 0:
            remaining_text_ids = list(set(dataset.global_text_ids) - set(text_ids_unique))
            remaining_text_ids = random.sample(remaining_text_ids, num_neg_texts)
            embeddings_to_assign = dataset.embeddings[remaining_text_ids]
            text_embeddings[num_unique_to_fill:,:] = embeddings_to_assign

    labels = torch.tensor([text2idx[text] for text in text_ids], dtype=torch.int64)
    text_ids_final = text_ids_unique + remaining_text_ids
    texts = [dataset.text_list[text_id] for text_id in text_ids_final]
    targets['instances']['bboxes'] = boxes
    targets['instances']['labels'] = labels
    targets['text_ids'] = text_ids_final
    targets['instances']['texts'] = texts
    targets['text_feats'] = text_embeddings
    return img, targets

def train_collate_cp(data_batch: Tuple[np.ndarray, Dict[str, Any]], n_objects=4) -> Dict:
    objects, _ = get_objects_from_batch(data_batch)
    dataset = data_batch[0][1]['dataset']
    # Pre-allocate lists for batch data
    batch_imgs = []
    batch_embeddings = []
    batch_bboxes_labels = []
    num_boxes_per_sample = []
    text_ids = []
    # Process each sample in the batch
    for i, sample in enumerate(data_batch):
        sample = apply_copy_paste(sample, objects, dataset, n_objects=n_objects)

        inputs = sample[0]
        data_instances = sample[1]['instances']
        embeddings = sample[1]['text_feats']

        batch_imgs.append(inputs)
        batch_embeddings.append(embeddings)
        gt_bboxes = data_instances['bboxes']
        gt_labels = data_instances['labels']
        batch_idx = torch.full((len(gt_labels), 1), i, device=gt_labels.device)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes), dim=1)
        batch_bboxes_labels.append(bboxes_labels)
        num_boxes_per_sample.append(len(gt_bboxes))
        text_ids.extend([sample[1]['text_ids'][j] for j in gt_labels.to(torch.int64)])

    # Construct collated results
    collated_results = {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': {
            'text_feats': torch.stack(batch_embeddings, 0),
            'idx': torch.cat(batch_bboxes_labels, 0)[:, 0].to(torch.int64),  # batch indices
            'labels': torch.cat(batch_bboxes_labels, 0)[:, 1].to(torch.int64),  # labels
            'num_boxes': num_boxes_per_sample,  # Number of boxes per sample
            'text_ids': text_ids
        }
    }
    # Process bounding boxes with optimized conversion
    boxes = torch.cat(batch_bboxes_labels, 0)[:, 2:]  # Extract bounding box coordinates (from second column)
    width, height = collated_results['inputs'].shape[2:][::-1]  # [width, height]
    norm_factor = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
    boxes = boxes / norm_factor
    boxes = torch.clamp(boxes, min=0.0, max=1.0)
    collated_results['data_samples']['boxes'] = boxes
    collated_results['data_samples']['norm_factor'] = norm_factor

    return collated_results

train_collate_cp_16 = partial(train_collate_cp, n_objects=16)

def apply_mixup(sample1, sample2, dataset, alpha=0.5):
    img, targets = copy.deepcopy(sample1[0]), sample1[1]
    mixup_img, mixup_targets = copy.deepcopy(sample2[0]), sample2[1]
    img = img * alpha + mixup_img * (1 - alpha)

    combined_instances = [targets['instances'], mixup_targets['instances']]
    
    text_ids_o = targets['text_ids'].copy()
    text_ids = [text_ids_o[i] for i in targets['instances']['labels'].to(torch.int64)]
    text_ids_o_mix = mixup_targets['text_ids'].copy()
    text_ids.extend([text_ids_o_mix[i] for i in mixup_targets['instances']['labels'].to(torch.int64)])
    
    text_ids_unique = list(set(text_ids))
    text2idx = {text_id_val: local_idx for local_idx, text_id_val in enumerate(text_ids_unique)}
    
    num_embedding_slots = targets['text_feats'].shape[0]
    placeholder_embedding = dataset.embeddings[dataset.text_to_index[" "]]
    text_embeddings = placeholder_embedding.repeat(num_embedding_slots, 1)
    num_unique_to_fill = min(len(text_ids_unique), num_embedding_slots)
    if num_unique_to_fill > 0:
        embeddings_to_assign = dataset.embeddings[text_ids_unique[:num_unique_to_fill]]
        text_embeddings[:num_unique_to_fill,:] = embeddings_to_assign
        num_neg_texts = num_embedding_slots - num_unique_to_fill
        if num_neg_texts > 0:
            remaining_text_ids = list(set(dataset.global_text_ids) - set(text_ids_unique))
            remaining_text_ids = random.sample(remaining_text_ids, num_neg_texts)
            embeddings_to_assign = dataset.embeddings[remaining_text_ids]
            text_embeddings[num_unique_to_fill:,:] = embeddings_to_assign
    else:
        remaining_text_ids = []
    
    labels = []
    bboxes = []
    for instance in combined_instances:
        for i, label in enumerate(instance['labels']):
            text_id_val = targets['text_ids'][int(label)] if instance is targets['instances'] else mixup_targets['text_ids'][int(label)]
            labels.append(text2idx[text_id_val])
        bboxes.append(instance['bboxes'])

    targets = {}
    targets['instances'] = {}
    targets['instances']['labels'] = torch.tensor(labels, dtype=torch.int64)
    targets['instances']['bboxes'] = torch.cat(bboxes, axis=0)
    
    text_ids_final = text_ids_unique + remaining_text_ids
    texts = [dataset.text_list[text_id] for text_id in text_ids_final]
    targets['instances']['texts'] = texts
    targets['text_ids'] = text_ids_final
    targets['text_feats'] = text_embeddings
    return img, targets

def train_collate_mixup(data_batch: Tuple[np.ndarray, Dict[str, Any]]) -> Dict:
    dataset = data_batch[0][1]['dataset']
    # Pre-allocate lists for batch data
    batch_imgs = []
    batch_embeddings = []
    batch_bboxes_labels = []
    num_boxes_per_sample = []
    text_ids = []
    # Process each sample in the batch
    for i, sample in enumerate(data_batch):
        sample = apply_mixup(sample, data_batch[i-1], dataset, alpha=0.5)
        inputs = sample[0]
        data_instances = sample[1]['instances']
        embeddings = sample[1]['text_feats']
        
        batch_imgs.append(inputs)
        batch_embeddings.append(embeddings)
        gt_bboxes = data_instances['bboxes']
        gt_labels = data_instances['labels']
        batch_idx = torch.full((len(gt_labels), 1), i, device=gt_labels.device)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes), dim=1)
        batch_bboxes_labels.append(bboxes_labels)
        num_boxes_per_sample.append(len(gt_bboxes))
        text_ids.extend([sample[1]['text_ids'][j] for j in gt_labels.to(torch.int64)])

    # Construct collated results
    collated_results = {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': {
            'text_feats': torch.stack(batch_embeddings, 0),
            'idx': torch.cat(batch_bboxes_labels, 0)[:, 0].to(torch.int64),  # batch indices
            'labels': torch.cat(batch_bboxes_labels, 0)[:, 1].to(torch.int64),  # labels
            'num_boxes': num_boxes_per_sample,  # Number of boxes per sample
            'text_ids': text_ids
        }
    }

    # Process bounding boxes with optimized conversion
    boxes = torch.cat(batch_bboxes_labels, 0)[:, 2:]  # Extract bounding box coordinates (from second column)
    width, height = collated_results['inputs'].shape[2:][::-1]  # [width, height]
    norm_factor = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
    boxes = boxes / norm_factor
    boxes = torch.clamp(boxes, min=0.0, max=1.0)
    collated_results['data_samples']['boxes'] = boxes
    collated_results['data_samples']['norm_factor'] = norm_factor
    
    return collated_results

def apply_grid_paste(sample, objects, dataset, mode_w = 4, mode_h=4,
                     alpha_fg = 0.5, alpha_bg = 0.5, 
                     ignore_anns = False, ignore_img = False, 
                     ignore_ori_img = True, ob_aug = True):
    side_length_w, side_length_h = int(640 / mode_w), int(640 / mode_h)
    img, targets = sample
    if ignore_img:
        img = torch.zeros_like(img)
    elif ignore_ori_img:
        img = targets['img_bg']

    selected_indices = random.choices(range(len(objects)), k=mode_w*mode_h)
    
    if ignore_anns:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        text_ids = []
    else:
        boxes = targets['instances']['bboxes'].clone()
        text_ids_o = targets['text_ids'].copy()
        text_ids = [text_ids_o[i] for i in targets['instances']['labels'].to(torch.int64)]
    
    for idx in range(mode_w * mode_h):
        row = idx // mode_w
        col = idx % mode_w
        y_offset = row * side_length_h
        x_offset = col * side_length_w

        img_crop_box, relative_coord, text_id = copy.deepcopy(objects[selected_indices[idx]])
        img_crop_box = img_crop_box.permute(1, 2, 0).numpy()
        relative_coord = relative_coord.unsqueeze(0).numpy()
        
        # object-level augmentation
        if ob_aug:
            img_crop_box, relative_coord = random_horizontal_flip(img_crop_box, relative_coord)
            if img_crop_box.shape[0] >= side_length_h or img_crop_box.shape[1] >= side_length_w:
                img_crop_box, relative_coord = random_iou_crop(img_crop_box, relative_coord)
        
        # resize to fit grid
        if img_crop_box.shape[0] > side_length_h or img_crop_box.shape[1] > side_length_w:
            if mode_h == 4 and mode_w == 4:
                img_crop_box, relative_coord = object_resize_44(img_crop_box, relative_coord)
            elif mode_h == 8 and mode_w == 4:
                img_crop_box, relative_coord = object_resize_84(img_crop_box, relative_coord)
            elif mode_h == 4 and mode_w == 8:
                img_crop_box, relative_coord = object_resize_48(img_crop_box, relative_coord)
            elif mode_h == 8 and mode_w == 8:
                img_crop_box, relative_coord = object_resize_88(img_crop_box, relative_coord)   
            elif mode_h == 2 and mode_w ==2:
                img_crop_box, relative_coord = object_resize_22(img_crop_box, relative_coord)
            elif mode_h == 1 and mode_w ==2:    
                img_crop_box, relative_coord = object_resize_12(img_crop_box, relative_coord)
            elif mode_h == 2 and mode_w ==1:    
                img_crop_box, relative_coord = object_resize_21(img_crop_box, relative_coord)
            elif mode_h == 1 and mode_w ==1:    
                img_crop_box, relative_coord = object_resize_11(img_crop_box, relative_coord)              

        img_crop_box = torch.from_numpy(img_crop_box).permute(2, 0, 1)
        rel_x1, rel_y1, rel_x2, rel_y2 = relative_coord[0]

        max_y_offset = max(0, side_length_h - img_crop_box.shape[1])
        max_x_offset = max(0, side_length_w - img_crop_box.shape[2])
        rand_y = random.randint(0, max_y_offset)
        rand_x = random.randint(0, max_x_offset)

        y_start = y_offset + rand_y
        x_start = x_offset + rand_x
        y_end = y_start + img_crop_box.shape[1]
        x_end = x_start + img_crop_box.shape[2]

        abs_x1 = x_start + rel_x1
        abs_y1 = y_start + rel_y1
        abs_x2 = x_start + rel_x2
        abs_y2 = y_start + rel_y2

        mask = torch.zeros((1, img_crop_box.shape[1], img_crop_box.shape[2]), dtype=torch.bool)
        mask[:, int(rel_y1):int(rel_y2), int(rel_x1):int(rel_x2)] = True

        ori_img = img[:, y_start:y_end, x_start:x_end]
        mixed_img = ori_img * (~mask) + img_crop_box * mask * alpha_fg + ori_img * mask * (1 - alpha_fg)
        background_mask = ~mask
        mixed_img[background_mask.expand_as(mixed_img)] = img_crop_box[background_mask.expand_as(img_crop_box)] * alpha_bg + ori_img[background_mask.expand_as(ori_img)] * (1 - alpha_bg)

        img[:, y_start:y_end, x_start:x_end] = mixed_img

        new_boxes = torch.tensor([[abs_x1, abs_y1, abs_x2, abs_y2]], dtype=torch.float32)
        boxes = torch.cat([boxes, new_boxes], dim=0)
        text_ids.append(text_id)
    text_ids_unique = list(set(text_ids))
    text2idx = {text_id_val: local_idx for local_idx, text_id_val in enumerate(text_ids_unique)}
    num_embedding_slots = targets['text_feats'].shape[0]
    placeholder_embedding = dataset.embeddings[dataset.text_to_index[" "]]
    text_embeddings = placeholder_embedding.repeat(num_embedding_slots, 1)
    num_unique_to_fill = min(len(text_ids_unique), num_embedding_slots)
    if num_unique_to_fill > 0:
        embeddings_to_assign = dataset.embeddings[text_ids_unique[:num_unique_to_fill]]
        text_embeddings[:num_unique_to_fill,:] = embeddings_to_assign
        num_neg_texts = num_embedding_slots - num_unique_to_fill
        if num_neg_texts > 0:
            remaining_text_ids = list(set(dataset.global_text_ids) - set(text_ids_unique))
            remaining_text_ids = random.sample(remaining_text_ids, num_neg_texts)
            embeddings_to_assign = dataset.embeddings[remaining_text_ids]
            text_embeddings[num_unique_to_fill:,:] = embeddings_to_assign
    labels = torch.tensor([text2idx[text] for text in text_ids], dtype=torch.int64)
    text_ids_final = text_ids_unique + remaining_text_ids
    texts = [dataset.text_list[text_id] for text_id in text_ids_final]
    targets['instances']['bboxes'] = boxes
    targets['instances']['labels'] = labels
    targets['text_ids'] = text_ids_final
    targets['instances']['texts'] = texts
    targets['text_feats'] = text_embeddings
    return img, targets

def train_collate_gridsynthetic(data_batch: Tuple[np.ndarray, Dict[str, Any]], 
                             mode_w_list = [4,8], mode_h_list=[4,8], alpha_fg = 1.0, alpha_bg = 1.0, 
                             ignore_anns = True, ignore_img = True, 
                             ignore_ori_img = True, ob_aug = True, css = True) -> Dict:
    objects, objects_L = get_objects_from_batch(data_batch)
    dataset = data_batch[0][1]['dataset']
    # Pre-allocate lists for batch data
    batch_imgs = []
    batch_embeddings = []
    batch_bboxes_labels = []
    num_boxes_per_sample = []
    text_ids = []
    # Process each sample in the batch
    for i, sample in enumerate(data_batch):
        alpha_bg = 1.0
        mode_w, mode_h = random.choice(mode_w_list), random.choice(mode_h_list)
        sample = apply_grid_paste(sample, objects, dataset, 
                                  mode_w = mode_w, mode_h=mode_h, alpha_fg = alpha_fg, 
                                  alpha_bg = alpha_bg, 
                                  ignore_anns = ignore_anns,
                                  ignore_img = ignore_img,
                                  ignore_ori_img = ignore_ori_img,
                                  ob_aug = ob_aug)
        if random.random() < 0.5 and css:
            # Complex Scene Simulation, CSS
            alpha_bg = 0.5 if alpha_bg == 1.0 else 0.0
            if len(objects_L) > 0 and random.random() < 0.5:
                mode_w_l, mode_h_l = random.choice([1, 2]), random.choice([1, 2])
                sample = apply_grid_paste(sample, objects_L, dataset, 
                                        mode_w = mode_w_l, mode_h=mode_h_l, alpha_fg = 0.5, 
                                        alpha_bg = alpha_bg, 
                                        ignore_anns = False,
                                        ignore_img = False,
                                        ignore_ori_img = False,
                                        ob_aug = ob_aug)
            else:
                mode_w, mode_h = random.choice(mode_w_list), random.choice(mode_h_list)
                sample = apply_grid_paste(sample, objects, dataset, 
                                   mode_w = mode_w, mode_h=mode_h, alpha_fg = 0.5, 
                                   alpha_bg = alpha_bg, 
                                   ignore_anns = False,
                                   ignore_img = False,
                                   ignore_ori_img = False,
                                   ob_aug = ob_aug)
        inputs = sample[0]
        data_instances = sample[1]['instances']
        embeddings = sample[1]['text_feats']
        
        batch_imgs.append(inputs)
        batch_embeddings.append(embeddings)
        gt_bboxes = data_instances['bboxes']
        gt_labels = data_instances['labels']
        batch_idx = torch.full((len(gt_labels), 1), i, device=gt_labels.device)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes), dim=1)
        batch_bboxes_labels.append(bboxes_labels)
        num_boxes_per_sample.append(len(gt_bboxes))
        text_ids.extend([sample[1]['text_ids'][j] for j in gt_labels.to(torch.int64)])

    # Construct collated results
    collated_results = {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': {
            'text_feats': torch.stack(batch_embeddings, 0),
            'idx': torch.cat(batch_bboxes_labels, 0)[:, 0].to(torch.int64),  # batch indices
            'labels': torch.cat(batch_bboxes_labels, 0)[:, 1].to(torch.int64),  # labels
            'num_boxes': num_boxes_per_sample,  # Number of boxes per sample
            'text_ids': text_ids
        }
    }

    # Process bounding boxes with optimized conversion
    boxes = torch.cat(batch_bboxes_labels, 0)[:, 2:]  # Extract bounding box coordinates (from second column)
    width, height = collated_results['inputs'].shape[2:][::-1]  # [width, height]
    norm_factor = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')
    boxes = boxes / norm_factor
    boxes = torch.clamp(boxes, min=0.0, max=1.0)
    collated_results['data_samples']['boxes'] = boxes
    collated_results['data_samples']['norm_factor'] = norm_factor
    
    return collated_results

def train_collate_final(data_batch):
    if random.random() < 0.5:
        collated_results = train_collate_gridsynthetic(data_batch)
    else:
        collated_results = train_collate_mixup(data_batch)
    return collated_results

train_collate_gridsynthetic_w_img = partial(train_collate_gridsynthetic,
                                            ignore_img = False, ignore_ori_img = False)

train_collate_gridsynthetic_wo_expansion = partial(train_collate_gridsynthetic,
                                                    alpha_bg = 0.0)

train_collate_gridsynthetic_wo_css = partial(train_collate_gridsynthetic,
                                            css=False)


def eval_collate(data_batch: Tuple[np.ndarray, Dict[str, Any]]) -> Dict:
    # Pre-allocate lists for batch data
    batch_imgs = []
    batch_embeddings = []
    img_ids = []
    ori_shapes = []
    batch_bboxes_labels = []
    num_boxes_per_sample = []
    pad_params = []
    scale_factors = []

    # Process each sample in the batch
    for i, sample in enumerate(data_batch):
        inputs = sample[0]
        data_instances = sample[1]['instances']
        embeddings = sample[1]['text_feats']
        
        # Collect images
        batch_embeddings.append(embeddings)
        batch_imgs.append(inputs)
        # Process bounding boxes and labels
        img_ids.append(data_instances['img_id'])
        ori_shapes.append(torch.tensor(data_instances['ori_shape']))  # [height, width]
        gt_bboxes = data_instances['bboxes']
        gt_labels = data_instances['labels']
        scale_factor = torch.tensor(sample[1]['scale_factor'], dtype=torch.float32)
        pad_param = sample[1]['pad_param']
        scale_factors.append(scale_factor)
        pad_params.append(pad_param)
        
        batch_idx = torch.full((len(gt_labels), 1), i, device=gt_labels.device)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes), dim=1)
        batch_bboxes_labels.append(bboxes_labels)
        
        # Store number of boxes for this sample
        num_boxes_per_sample.append(len(gt_bboxes))
    # Construct collated results
    collated_results = {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': {
            'text_feats':torch.stack(batch_embeddings, 0),
            'idx': torch.cat(batch_bboxes_labels, 0)[:, 0],  # batch indices
            'labels': torch.cat(batch_bboxes_labels, 0)[:, 1],  # labels
            'num_boxes': num_boxes_per_sample,  # Add num_boxes,
            'pad_params':torch.stack(pad_params,0),
            'scale_factors':torch.stack(scale_factors,0),
            'ori_shapes': torch.stack(ori_shapes,0)
        }
    }
    boxes = torch.cat(batch_bboxes_labels, 0)[:, 2:]  # Extract bounding box coordinates (from second column)
    boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')
    collated_results['data_samples']['boxes'] = boxes
    collated_results['data_samples']['img_id'] = img_ids
    
    return collated_results
