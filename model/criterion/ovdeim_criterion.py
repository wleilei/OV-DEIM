"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision
import copy
from ..decoder.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .matcher import HungarianMatcher

class OVDEIMCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]

    def __init__(self, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False,
        matcher_args = {
            'weight_dict': {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
            'alpha': 0.25, 'gamma': 2.0
            }):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(**matcher_args)
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def _get_indices(self, indices, targets):
        idx = self._get_src_permutation_idx(indices)  # (batch_idx, src_idx)
        offsets = torch.cumsum(torch.tensor([0] + targets['num_boxes'][:-1]), dim=0)
        target_indices = torch.cat([j + offsets[i] for i, (_, j) in enumerate(indices)], dim=0)
        return idx, target_indices

    def loss_labels_focal(self, outputs, targets, num_boxes, idx, target_indices):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        target_classes_o = targets['labels'][target_indices].to(dtype=torch.int64)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.sum(2) / num_boxes  # (B, Q)
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, num_boxes, idx, target_indices, values=None):
        assert 'pred_boxes' in outputs
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = targets['boxes'][target_indices]
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach() # [M]
        else:
            ious = values

        src_logits = outputs['pred_logits'] # [B, Q, C]
        target_classes_o = targets['labels'][target_indices].to(dtype=torch.int64) # [M]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # [B, Q]
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1] # [B, Q, C]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score  
              
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, 
                                                  reduction='none')
        
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # (B, Q) if dim=2
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, num_boxes, idx, target_indices, values=None):
        assert 'pred_boxes' in outputs
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = targets['boxes'][target_indices]
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = targets['labels'][target_indices].to(dtype=torch.int64)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # (B, Q) if dim=2
        return {'loss_mal': loss}
    
    def loss_boxes(self, outputs, targets, num_boxes, idx, target_indices, boxes_weight=None):
        assert 'pred_boxes' in outputs
        bs, num_queries = outputs['pred_boxes'].shape[:2]
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = targets['boxes'][target_indices]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')  # [num_matched_boxes, 4]
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))  # [num_matched_boxes]
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight

        loss_bbox_per_query = torch.zeros([bs, num_queries, 4], device=src_boxes.device)
        loss_bbox_per_query[idx] = loss_bbox
        loss_bbox = loss_bbox_per_query.sum() / num_boxes  # [B, Q] if dim=2

        loss_giou_per_query = torch.zeros([bs, num_queries], device=src_boxes.device)
        loss_giou_per_query[idx] = loss_giou
        loss_giou = loss_giou_per_query.sum() / num_boxes  # [B, Q] if dim=2

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
        
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, num_boxes, idx, target_indices, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, idx, target_indices, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        num_boxes = sum(targets['num_boxes'])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float32, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
            num_boxes = torch.clamp(num_boxes / world_size, min=1).item()
        else:
            num_boxes = min(num_boxes.item(), 1.0)
        
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']
        idx, target_indices = self._get_indices(indices, targets)
        # idx_dec, target_indices_dec = idx, target_indices

        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, idx, target_indices)
            l_dict = self.get_loss(loss, outputs, targets, num_boxes, idx, target_indices, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                    idx, target_indices = self._get_indices(indices, targets)
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, idx, target_indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, idx, target_indices, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            idx, target_indices = self._get_indices(indices, targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, idx, target_indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, dn_num_boxes, idx, target_indices, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                idx, target_indices = self._get_indices(indices, enc_targets)
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, idx, target_indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, num_boxes, idx, target_indices, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, idx, target_indices):
        if self.boxes_weight_format is None:
            return {}
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = targets['boxes'][target_indices]
        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}
        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = targets['num_boxes']
        device = targets['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                                        torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices