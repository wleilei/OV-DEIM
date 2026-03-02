"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(data_samples,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None

    idx = data_samples['idx'] 
    labels = data_samples['labels'] 
    boxes = data_samples['boxes'] 
    device = labels.device
    num_gts = data_samples['num_boxes'] 
    max_gt_num = max(num_gts)
    batch_size = len(num_gts)

    if max_gt_num == 0:
        return None, None, None, None

    num_group = max(1, num_denoising // max_gt_num)

    input_query_class = torch.full([batch_size, max_gt_num], num_classes, 
                                 dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([batch_size, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([batch_size, max_gt_num], dtype=torch.bool, device=device)

    for i in range(batch_size):
        mask = (idx == i)
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = labels[mask]
            input_query_bbox[i, :num_gt] = boxes[mask]
            pad_gt_mask[i, :num_gt] = 1

    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    dn_positive_idx = []
    for i in range(batch_size):
        num_gt = num_gts[i]
        indices = torch.cat([torch.arange(num_gt, device=device) + k * max_gt_num * 2 
                             for k in range(num_group)])
        dn_positive_idx.append(indices)

    num_denoising = max_gt_num * 2 * num_group

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5) # 0.5 * 0.5 = 0.25
        new_label = torch.randint(0, num_classes, mask.shape, device=device)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = (input_query_bbox[..., 2:] * 0.5).repeat(1, 1, 2) * box_noise_scale
        rand_sign = torch.randint(0, 2, input_query_bbox.shape, device=device) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        negative_gt_mask = torch.zeros([batch_size, max_gt_num * 2 * num_group, 4], device=device)
        for k in range(num_group):
            start_idx = k * max_gt_num * 2 + max_gt_num
            end_idx = (k + 1) * max_gt_num * 2
            negative_gt_mask[:, start_idx:end_idx, :] = 1
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += (rand_sign * rand_part * diff)
        known_bbox = torch.clamp(known_bbox, 0.0, 1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
    else:
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_embed = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True 

    for i in range(num_group):
        start = max_gt_num * 2 * i
        end = max_gt_num * 2 * (i + 1)
        if i == 0:
            attn_mask[start:end, end:num_denoising] = True
        elif i == num_group - 1:
            attn_mask[start:end, :start] = True
        else:
            attn_mask[start:end, :start] = True
            attn_mask[start:end, end:num_denoising] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_embed, input_query_bbox_unact, attn_mask, dn_meta


