"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def mod(a, b):
    out = a - a // b * b
    return out

def box_convert(boxes, in_fmt='cxcywh', out_fmt='xywh'):
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes to have shape (B, N, 4), got {boxes.shape}")

    if in_fmt == 'cxcywh' and out_fmt == 'xywh':
        cx = boxes[:, :, 0] 
        cy = boxes[:, :, 1]  
        w = boxes[:, :, 2]  
        h = boxes[:, :, 3]  

        x = cx - w / 2  
        y = cy - h / 2  

        converted_boxes = torch.stack([x, y, w, h], dim=2)
        return converted_boxes

    elif in_fmt == 'cxcywh' and out_fmt == 'xyxy':
        cx = boxes[:, :, 0]  
        cy = boxes[:, :, 1]  
        w = boxes[:, :, 2]   
        h = boxes[:, :, 3]  

        x_min = cx - w / 2 
        y_min = cy - h / 2  
        x_max = cx + w / 2  
        y_max = cy + h / 2  

        converted_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=2) 
        return converted_boxes
    elif in_fmt == 'xyxy' and out_fmt == 'xywh':
        x_min = boxes[:, :, 0]  
        y_min = boxes[:, :, 1] 
        x_max = boxes[:, :, 2] 
        y_max = boxes[:, :, 3]  

        w = x_max - x_min  
        h = y_max - y_min  
        x = x_min
        y = y_min
        converted_boxes = torch.stack([x, y, w, h], dim=2) 
        return converted_boxes

    else:
        raise ValueError(f"Unsupported conversion from {in_fmt} to {out_fmt}")
    
def restore_pred_boxes(boxes, img_shape, target):
    h_img, w_img = img_shape
    scale_w, scale_h = target['scale_factors'].split(1, dim=1)
    top_pad,_, left_pad,_ = target['pad_params'].split(1,dim=1)

    norm_factor = torch.tensor([w_img, h_img, w_img, h_img], device=boxes.device)
    boxes = box_convert(boxes * norm_factor, in_fmt='cxcywh', out_fmt='xyxy')

    batch_size, num_boxes = boxes.shape[:2]
    device = boxes.device

    left_pad_exp = left_pad.unsqueeze(1).expand(batch_size, num_boxes, 1).to(device)
    top_pad_exp = top_pad.unsqueeze(1).expand(batch_size, num_boxes, 1).to(device)
    scale_w_exp = scale_w.unsqueeze(1).expand(batch_size, num_boxes, 1).to(device)
    scale_h_exp = scale_h.unsqueeze(1).expand(batch_size, num_boxes, 1).to(device)

    pad_adjust = torch.cat([left_pad_exp, top_pad_exp, left_pad_exp, top_pad_exp], dim=-1)
    scale_adjust = torch.cat([scale_w_exp, scale_h_exp, scale_w_exp, scale_h_exp], dim=-1)

    boxes.sub_(pad_adjust)
    boxes.div_(scale_adjust)

    ori_shapes = target['ori_shapes']  # [16, 2]
    ori_heights, ori_widths = ori_shapes[:, 0], ori_shapes[:, 1]
    max_x = ori_widths.unsqueeze(1).expand(batch_size, num_boxes).to(device)  # [16, num_boxes]
    max_y = ori_heights.unsqueeze(1).expand(batch_size, num_boxes).to(device) # [16, num_boxes]

    min_tensor = torch.zeros_like(max_x) 

    boxes[..., 0] = boxes[..., 0].clamp(min=min_tensor, max=max_x)  # x_min
    boxes[..., 1] = boxes[..., 1].clamp(min=min_tensor, max=max_y)  # y_min
    boxes[..., 2] = boxes[..., 2].clamp(min=min_tensor, max=max_x)  # x_max
    boxes[..., 3] = boxes[..., 3].clamp(min=min_tensor, max=max_y)  # y_max
    
    boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')
    
    return boxes


class OVDEIMPostProcessor(nn.Module):
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        image_shape = (640,640)
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.image_shape = image_shape

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    def forward(self, outputs, target):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']    
        bbox_pred = restore_pred_boxes(boxes, self.image_shape, target)
        
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            gather_index = index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            boxes = bbox_pred.gather(dim=1, index=gather_index)
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(bbox_pred, dim=1, index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]))
        
        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results

