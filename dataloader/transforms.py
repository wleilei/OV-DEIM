import math
import random
from typing import Any, Dict, List, Tuple, Sequence, Optional, Union
import torch.nn as nn
import cv2
import json
import numpy as np
import albumentations as A
import torch

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        for transform in self.transforms:
            data = transform(data)  
        return data
    
def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    return intersection / union

class RandomIoUCrop(nn.Module):
    def __init__(self,
                 min_scale: float = 0.5,
                 max_scale: float = 1.0,
                 min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0,
                 min_iou: float = 0.3,
                 trials: int = 10,
                 p: float = 0.5):
        """
        Implements random IoU crop data augmentation
        
        Args:
            min_scale: Minimum scale ratio for cropped image (0.3)
            max_scale: Maximum scale ratio for cropped image (1.0)
            min_aspect_ratio: Minimum aspect ratio for crop region (0.5)
            max_aspect_ratio: Maximum aspect ratio for crop region (2.0)
            min_iou: Minimum IoU threshold (0.3)
            trials: Number of crop attempts (10)
            p: Probability of applying this augmentation (0.5)
        """
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_iou = min_iou
        self.trials = trials
        self.p = p

    def forward(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        if random.random() > self.p:
            return img, bboxes
        
        if bboxes is None or len(bboxes) == 0:
            return img, bboxes
        
        h, w = img.shape[:2]        
        for _ in range(self.trials):
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect_ratio = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            
            crop_w = int(w * scale)
            crop_h = int(h * scale)
            
            if aspect_ratio > 1:
                crop_h = int(crop_w / aspect_ratio)
            else:
                crop_w = int(crop_h * aspect_ratio)
            
            if crop_w > w or crop_h > h:
                continue
            
            if crop_w <= 0 or crop_h <= 0:
                continue
            
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            
            ious = []
            for bbox in bboxes:
                crop_box = np.array([x1, y1, x2, y2])
                iou = compute_iou(bbox, crop_box)
                ious.append(iou)
            
            if min(ious) >= self.min_iou:
                cropped_img = img[y1:y2, x1:x2]
                
                cropped_bboxes = bboxes.copy()
                cropped_bboxes[:, 0] = np.clip(cropped_bboxes[:, 0] - x1, 0, crop_w)
                cropped_bboxes[:, 1] = np.clip(cropped_bboxes[:, 1] - y1, 0, crop_h)
                cropped_bboxes[:, 2] = np.clip(cropped_bboxes[:, 2] - x1, 0, crop_w)
                cropped_bboxes[:, 3] = np.clip(cropped_bboxes[:, 3] - y1, 0, crop_h)
                
                valid_indices = np.where((cropped_bboxes[:, 0] < crop_w) & 
                                        (cropped_bboxes[:, 1] < crop_h) & 
                                        (cropped_bboxes[:, 2] > 0) & 
                                        (cropped_bboxes[:, 3] > 0))[0]
                cropped_bboxes = cropped_bboxes[valid_indices]
                
                if len(cropped_bboxes) == 0:
                    continue
                
                return cropped_img, cropped_bboxes
        
        return img, bboxes

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        img: np.ndarray,
        bboxes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() > self.p:
            return img, bboxes

        h, w = img.shape[:2]

        flipped_img = img[:, ::-1, :].copy()

        if bboxes is None or len(bboxes) == 0:
            return flipped_img, bboxes

        flipped_bboxes = bboxes.copy()

        x1 = bboxes[:, 0]
        x2 = bboxes[:, 2]

        flipped_bboxes[:, 0] = w - x2
        flipped_bboxes[:, 2] = w - x1

        return flipped_img, flipped_bboxes

    
class ObjectResize(nn.Module):
    def __init__(self, 
                 size: Tuple[int, int],
                 pad_val: dict = dict(img=114, mask=0, seg=255),
                 allow_scale_up: bool = True,
                 interpolation: str = 'bilinear'):
        super().__init__()
        self.size = size
        self.pad_val = pad_val
        self.allow_scale_up = allow_scale_up
        self.interpolation = cv2_interp_codes[interpolation]
    
    def forward(self, 
                img: np.ndarray, 
                bboxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img, bboxes = self._keep_ratio_resize(img, bboxes)
        img, bboxes = self._letterbox_resize(img, bboxes)
        return img, bboxes
    
    def _keep_ratio_resize(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = img.shape[:2]

        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: height={h}, width={w}")
        
        scale_factor = min(max(self.size) / max(h, w),
                         min(self.size) / min(h, w))
        
        new_w = int(np.rint(w * scale_factor))
        new_h = int(np.rint(h * scale_factor))
        
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        interp = self.interpolation if scale_factor >= 1 else cv2_interp_codes['area']
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        resized_bboxes = None
        if bboxes is not None:
            resized_bboxes = bboxes.copy().astype(np.float32)
            resized_bboxes *= scale_factor
            resized_bboxes = np.clip(resized_bboxes, 
                                   a_min=[0, 0, 0, 0], 
                                   a_max=[new_w, new_h, new_w, new_h])
        
        return resized_img, resized_bboxes
    
    def _letterbox_resize(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None, current_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if current_size is not None:
            h, w = current_size
        else:
            h, w = img.shape[:2]
            
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: height={h}, width={w}")
            
        target_h, target_w = self.size
        
        ratio = min(target_h / h, target_w / w)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)
        
        new_h = int(np.rint(ratio * h))
        new_w = int(np.rint(ratio * w))
        
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        if [h, w] != [new_h, new_w]:
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
        else:
            resized_img = img.copy()
        
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top = int(np.rint(pad_h / 2))
        left = int(np.rint(pad_w / 2))
        bottom = pad_h - top
        right = pad_w - left
        
        padding = [top, bottom, left, right]
        
        if any(padding):
            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and resized_img.ndim == 3:
                pad_val = tuple(pad_val for _ in range(resized_img.shape[2]))
            resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                           cv2.BORDER_CONSTANT, value=pad_val)
        
        resized_bboxes = None
        if bboxes is not None:
            resized_bboxes = bboxes.copy().astype(np.float32)
            if [h, w] != [new_h, new_w]:
                resized_bboxes[:, [0, 2]] *= new_w / w  
                resized_bboxes[:, [1, 3]] *= new_h / h  
            resized_bboxes[:, [0, 2]] += left  
            resized_bboxes[:, [1, 3]] += top   
            resized_bboxes[:, [0, 2]] = np.clip(resized_bboxes[:, [0, 2]], 0, target_w)
            resized_bboxes[:, [1, 3]] = np.clip(resized_bboxes[:, [1, 3]], 0, target_h)
        
        return resized_img, resized_bboxes

class KeepRatioResize(nn.Module):
    '''This code is adapted from the YOLOv5KeepRatioResize class in MMYolo:
    https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/transforms/transforms.py
    
    This class implements aspect ratio preserving resizing, which resizes an image to fit within a target size
    while maintaining its original aspect ratio. The resizing process:
    1. Calculates a scale factor based on the target size and original image dimensions
    2. Resizes the image using this scale factor
    3. Updates any bounding box annotations to match the new image size
    
    The scale factor is chosen as the minimum of:
    - max_long_edge / max(original height, original width)
    - max_short_edge / min(original height, original width)
    
    This ensures the resized image fits within the target dimensions while preserving aspect ratio.
    
    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords (optional)
        
    Output format: Same as input, but with resized img and updated targets:
        - targets['width'], targets['height']: updated to new dimensions
        - targets['scale_factor_wh']: (scale_w, scale_h) tuple
        - targets['instances']['bboxes']: updated and clipped if present
        
    Args:
        size (tuple[int, int]): Target size (height, width) to resize to
    '''
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        scale_factor = min(max(self.size) / max(targets['height'], targets['width']),
                         min(self.size) / min(targets['height'], targets['width']))
        
        new_w = int(np.rint(targets['width'] * scale_factor))
        new_h = int(np.rint(targets['height'] * scale_factor))
        
        img = cv2.resize(img, (new_w, new_h), 
                        interpolation=cv2_interp_codes['area' if scale_factor < 1 else 'bilinear'])
        
        if 'bboxes' in targets['instances']:
            targets['instances']['bboxes'] = np.clip(
                targets['instances']['bboxes'] * scale_factor,
                a_min=[0, 0, 0, 0],
                a_max=[new_w, new_h, new_w, new_h]
            )
            
        targets['scale_factor_wh'] = (new_w / targets['width'], new_h / targets['height'])
        targets['width'], targets['height'] = new_w, new_h
        
        return img, targets
    
class LetterResize(nn.Module):
    '''This code is adapted from the YOLOv5LetterResize class in MMYolo:
    https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/transforms/transforms.py
    
    This class implements letterbox resizing, which resizes an image to a target size while maintaining its aspect ratio.
    The resizing is done by:
    1. Scaling the image to fit within the target size while preserving aspect ratio
    2. Padding the scaled image with a constant value to reach the exact target size
    
    The padding is added equally on both sides (left/right for width, top/bottom for height).
    This approach is commonly used in object detection to avoid distorting the image content.
    
    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords (optional)
        targets['instances']['labels']: (N,) np.int64 (optional)
            
    Output format: Same as input, but with resized and padded img and updated targets:
        - targets['width'], targets['height']: updated to new dimensions
        - targets['scale_factor']: (scale_x, scale_y) tuple
        - targets['pad_param']: [top, bottom, left, right] padding values
        - targets['instances']['bboxes']: updated, shifted and clipped if present
        
    Args:
        size (tuple[int, int]): Target size (height, width)
        pad_val (dict): Padding values for different data types:
            - img: Value to pad image (default: 144)
            - mask: Value to pad masks (default: 0)
            - seg: Value to pad segmentation maps (default: 255)
        allow_scale_up (bool): Whether to allow scaling up the image. If False,
            the image will only be scaled down to fit the target size. Defaults to True.
    '''
    def __init__(self, size: Tuple[int, int], # hw
                 pad_val: dict = dict(img=114, mask=0, seg=255),
                 allow_scale_up: bool = True,):
        super().__init__()
        self.size = size
        self.pad_val = pad_val
        self.allow_scale_up = allow_scale_up

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        ratio = min(self.size[0] / targets['height'], self.size[1] / targets['width'])
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)
            
        new_h = int(np.rint(ratio * targets['height']))
        new_w = int(np.rint(ratio * targets['width']))
        
        if [targets['height'], targets['width']] != [new_h, new_w]:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2_interp_codes['bilinear'])
            
        targets['scale_factor'] = (new_w / targets['width'], new_h / targets['height'])
        
        # Calculate padding with stable rounding
        pad_h = self.size[0] - new_h
        pad_w = self.size[1] - new_w
        top = int(np.rint(pad_h / 2))
        left = int(np.rint(pad_w / 2))
        bottom = pad_h - top
        right = pad_w - left
        
        padding = [top, bottom, left, right]
        
        if any(padding):
            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and img.ndim == 3:
                pad_val = tuple(pad_val for _ in range(img.shape[2]))
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=pad_val)
            
            if 'bboxes' in targets['instances']:
                bboxes = targets['instances']['bboxes']
                bboxes[:, 0] *= new_w / targets['width']  # x1 * scale_x
                bboxes[:, 1] *= new_h / targets['height']  # y1 * scale_y
                bboxes[:, 2] *= new_w / targets['width']  # x2 * scale_x
                bboxes[:, 3] *= new_h / targets['height']  # y2 * scale_y
                bboxes[:, [0, 2]] += left
                bboxes[:, [1, 3]] += top
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img.shape[1])
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img.shape[0])
                targets['instances']['bboxes'] = bboxes
            
        targets['pad_param'] = np.array(padding, dtype=np.float32)
        targets['height'], targets['width'] = img.shape[:2]
        if 'scale_factor_wh' in targets:
            scale_factor_wh = targets.pop('scale_factor_wh')
            targets['scale_factor'] = (targets['scale_factor'][0] *
                                       scale_factor_wh[0],
                                       targets['scale_factor'][1] *
                                       scale_factor_wh[1])        
        return img, targets
    
class RandomAffine(nn.Module):
    """
    This code is adapted from the YOLOv5RandomAffine class in MMYolo:
    https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/transforms/transforms.py
    Manual implementation of Random affine transform similar to YOLOv5/v8.
    Applies rotation, translation, scaling, and shear using OpenCV.
    Filters bounding boxes based on size, aspect ratio, and area ratio.

    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords
        targets['instances']['labels']: (N,) np.int64 (optional)

    Output format: Same as input, but transformed img and targets. Bboxes are clipped.

    Args:
        max_rotate_degree (float): Max degrees rotation. Defaults to 10.0.
        max_translate_ratio (float): Max translation ratio (relative to height/width). Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min/max scaling factor. Defaults to (0.5, 1.5).
        max_shear_degree (float): Max shear degrees. Defaults to 2.0.
        border (tuple[int]): Extra border size (h, w) for output canvas. Used for mosaic. Defaults to (0, 0).
        border_val (tuple[int]): Border padding value (RGB). Defaults to (114, 114, 114).
        min_bbox_size (float): Min width/height for filtered bbox. Defaults to 1.0.
        min_area_ratio (float): Min transformed_area / original_area ratio. Defaults to 0.1.
        max_aspect_ratio (float): Max aspect ratio (w/h or h/w) for filtered bbox. Defaults to 20.0.
        bbox_clip_border (bool): Whether to clip bboxes to image boundaries after transform. Defaults to True.
    """
    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0), # (h, w) extra border
                 border_val: Tuple[int, int, int] = (114, 114, 114),
                 min_bbox_size: float = 1.0,
                 min_area_ratio: float = 0.1,
                 max_aspect_ratio: float = 20.0,
                 bbox_clip_border: bool = True):
        super().__init__()
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1] and scaling_ratio_range[0] > 0

        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border # Note: Original MMDet used (w, h), here using (h, w) for consistency
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        img_h, img_w = img.shape[:2]

        # Define output canvas size including border
        out_h = img_h + self.border[0] * 2
        out_w = img_w + self.border[1] * 2
        
        # Handle the case where border makes output smaller than input
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Border values too negative. Output size would be {out_h}x{out_w}")

        # --- 1. Calculate Random Affine Parameters ---
        # Rotation
        angle = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        # Scaling (uniform)
        scale = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        # Shear
        shear_x = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_y = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        # Translation (relative to output size)
        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * out_w
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * out_h

        # --- 2. Build transformation matrix using OpenCV style ---
        # Center of the input image
        center = (img_w / 2, img_h / 2)
        
        # Build the 2x3 affine matrix step by step
        # Start with rotation and scaling
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Add shear (we need to convert 2x3 to 3x3, add shear, then back)
        M_3x3 = np.vstack([M, [0, 0, 1]])
        
        # Shear matrix
        shear_matrix = np.array([[1, np.tan(np.radians(shear_x)), 0],
                                [np.tan(np.radians(shear_y)), 1, 0],
                                [0, 0, 1]], dtype=np.float32)
        
        # Apply shear
        M_3x3 = M_3x3 @ shear_matrix
        
        # Add translation and border offset
        M_3x3[0, 2] += trans_x + self.border[1]  # x translation + border offset
        M_3x3[1, 2] += trans_y + self.border[0]  # y translation + border offset
        
        # Convert back to 2x3 for cv2.warpAffine
        M = M_3x3[:2, :]

        # --- 3. Apply Transformation to Image ---
        img_transformed = cv2.warpAffine(img, M, dsize=(out_w, out_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=self.border_val)

        # Update image and dimensions in targets
        targets['height'] = out_h
        targets['width'] = out_w

        # --- 4. Transform and Filter Bounding Boxes ---
        if 'instances' in targets and 'bboxes' in targets['instances'] and targets['instances']['bboxes'].shape[0] > 0:
            original_bboxes = targets['instances']['bboxes'].copy()
            original_labels = targets['instances']['labels'].copy() if 'labels' in targets['instances'] else None

            # Transform bounding boxes using 3x3 matrix
            transformed_bboxes = self._transform_bboxes(original_bboxes, M_3x3, out_h, out_w)

            # Clip boxes first if required
            if self.bbox_clip_border:
                transformed_bboxes[:, [0, 2]] = np.clip(transformed_bboxes[:, [0, 2]], 0, out_w)
                transformed_bboxes[:, [1, 3]] = np.clip(transformed_bboxes[:, [1, 3]], 0, out_h)

            # Filter bounding boxes
            valid_indices = self._filter_bboxes(original_bboxes, transformed_bboxes, scale, out_h, out_w)

            # Keep only valid boxes and labels
            targets['instances']['bboxes'] = transformed_bboxes[valid_indices]
            if original_labels is not None:
                targets['instances']['labels'] = original_labels[valid_indices]
            else:
                targets['instances']['labels'] = np.empty((0,), dtype=np.int64)
            
            # Handle texts if present
            if 'texts' in targets['instances'] and len(targets['instances']['texts']) > 0:
                # For multimodal datasets, texts might need special handling
                pass
        
        elif 'instances' in targets:
            # Handle case with no input bboxes but instances key exists
            targets['instances']['bboxes'] = np.empty((0, 4), dtype=np.float32)
            targets['instances']['labels'] = np.empty((0,), dtype=np.int64)

        return img_transformed, targets

    def _transform_bboxes(self, bboxes_xyxy: np.ndarray, M: np.ndarray, h: int, w: int) -> np.ndarray:
        """Transforms bounding boxes using the affine matrix M."""
        n = bboxes_xyxy.shape[0]
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)

        # Convert boxes to corner points: (n, 4, 2) -> [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        corners = np.zeros((n, 4, 2), dtype=np.float32)
        corners[:, 0, :] = bboxes_xyxy[:, [0, 1]] # Top-left
        corners[:, 1, :] = bboxes_xyxy[:, [2, 1]] # Top-right
        corners[:, 2, :] = bboxes_xyxy[:, [2, 3]] # Bottom-right
        corners[:, 3, :] = bboxes_xyxy[:, [0, 3]] # Bottom-left

        # Reshape for matrix multiplication: (n*4, 2)
        corners = corners.reshape(-1, 2)

        # Add homogeneous coordinate: (n*4, 3)
        corners_hom = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)

        # Apply transformation matrix M (3x3)
        # corners_transformed_hom = corners_hom @ M.T # Incorrect order for cv2 warpPerspective matrix
        # Need to transpose M if defined as per cv2.warpPerspective convention
        corners_transformed_hom = corners_hom @ M.T

        # Normalize homogeneous coordinates (divide by z)
        z = corners_transformed_hom[:, 2:]
        # Avoid division by zero, replace zeros/small values with 1
        z[np.abs(z) < 1e-6] = 1.0
        corners_transformed = corners_transformed_hom[:, :2] / z

        # Reshape back to (n, 4, 2)
        corners_transformed = corners_transformed.reshape(n, 4, 2)

        # Find new axis-aligned bounding boxes (min/max coordinates)
        min_xy = np.min(corners_transformed, axis=1)
        max_xy = np.max(corners_transformed, axis=1)

        # Combine into [x1, y1, x2, y2] format
        transformed_bboxes_xyxy = np.concatenate([min_xy, max_xy], axis=1)

        return transformed_bboxes_xyxy

    def _filter_bboxes(self, orig_bboxes: np.ndarray, trans_bboxes: np.ndarray, scale: float, h: int, w: int) -> np.ndarray:
        """Filters bounding boxes based on size, aspect ratio, and area ratio."""
        if trans_bboxes.shape[0] == 0:
            return np.array([], dtype=np.int64)

        # Calculate dimensions of transformed boxes
        trans_w = trans_bboxes[:, 2] - trans_bboxes[:, 0]
        trans_h = trans_bboxes[:, 3] - trans_bboxes[:, 1]

        # Filter 1: Minimum size
        # Clip transformed boxes before size check? YOLOv5 doesn't explicitly, relies on area ratio maybe?
        # Let's check size before clipping first.
        size_valid = (trans_w >= self.min_bbox_size) & (trans_h >= self.min_bbox_size)

        # Filter 2: Area ratio
        orig_w = orig_bboxes[:, 2] - orig_bboxes[:, 0]
        orig_h = orig_bboxes[:, 3] - orig_bboxes[:, 1]
        orig_area = orig_w * orig_h
        trans_area = trans_w * trans_h
        # Avoid division by zero for degenerate original boxes
        area_valid = (orig_area < 1e-6) | ((trans_area / (orig_area + 1e-6)) >= self.min_area_ratio)

        # Filter 3: Aspect ratio
        # Avoid division by zero for degenerate transformed boxes
        aspect_ratio = np.maximum(trans_w / (trans_h + 1e-6), trans_h / (trans_w + 1e-6))
        aspect_ratio_valid = aspect_ratio < self.max_aspect_ratio

        # Combine filters
        valid_indices = np.where(size_valid & area_valid & aspect_ratio_valid)[0]
        return valid_indices

    # Helper static methods to generate transformation matrices (similar to user provided code)
    @staticmethod
    def _get_rotation_matrix(angle_degrees: float) -> np.ndarray:
        """Returns 3x3 rotation matrix."""
        radian = math.radians(angle_degrees)
        cos_a, sin_a = math.cos(radian), math.sin(radian)
        return np.array([[cos_a, -sin_a, 0],
                         [sin_a,  cos_a, 0],
                         [0,      0,     1]], dtype=np.float32)

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        """Returns 3x3 scaling matrix."""
        return np.array([[scale_ratio, 0,           0],
                         [0,           scale_ratio, 0],
                         [0,           0,           1]], dtype=np.float32)

    @staticmethod
    def _get_shear_matrix(x_degrees: float, y_degrees: float) -> np.ndarray:
        """Returns 3x3 shear matrix."""
        rad_x, rad_y = math.radians(x_degrees), math.radians(y_degrees)
        tan_x, tan_y = math.tan(rad_x), math.tan(rad_y)
        return np.array([[1,     tan_x, 0],
                         [tan_y, 1,     0],
                         [0,     0,     1]], dtype=np.float32)


class AlbumentationsTransform(nn.Module):
    '''Wrapper for applying Albumentations transformations to images and bounding boxes.
    
    This class provides integration between Albumentations transformations and the expected
    data format used in this codebase. It handles conversion between different bounding box
    formats and ensures proper updating of target metadata.
    
    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords (optional)
        targets['instances']['labels']: (N,) np.int64 (optional)
        
    Output format: Same as input, but with transformed img and updated targets:
        - targets['width'], targets['height']: updated to new dimensions
        - targets['instances']['bboxes']: transformed bboxes if present
        - targets['instances']['labels']: updated labels if present
        
    Args:
        transform: An albumentations transform or composition of transforms
    '''
    def __init__(self, transform):
        """
        Args:
            transform: An albumentations transform or composition of transforms
        """
        super().__init__()
        self.transform = transform
        
    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        
        # Extract bounding boxes if present
        bboxes = []
        labels = []
        if 'instances' in targets and 'bboxes' in targets['instances']:
            bboxes = targets['instances']['bboxes'].copy()
            if 'labels' in targets['instances']:
                labels = targets['instances']['labels'].copy()
        
        # Apply albumentations transform
        if len(bboxes) > 0:
            # Convert to albumentations format (normalized xyxy -> normalized xyxy)
            height, width = targets['height'], targets['width']
            albu_bboxes = bboxes.copy()
            albu_bboxes[:, [0, 2]] /= width  # normalize x coordinates
            albu_bboxes[:, [1, 3]] /= height  # normalize y coordinates
            
            # Clip bbox coordinates to valid range [0, 1] to handle floating point precision issues
            albu_bboxes = np.clip(albu_bboxes, 0.0, 1.0)
            
            # Apply transform with bounding boxes
            transformed = self.transform(
                image=img,
                bboxes=albu_bboxes,
                labels=labels,
            )
            
            # Update image and bounding boxes
            img = transformed['image']
            if 'bboxes' in transformed and len(transformed['bboxes']) > 0:
                # Convert back from albumentations format
                transformed_bboxes = np.array(transformed['bboxes'])
                transformed_bboxes[:, [0, 2]] *= img.shape[1]  # denormalize x
                transformed_bboxes[:, [1, 3]] *= img.shape[0]  # denormalize y
                targets['instances']['bboxes'] = transformed_bboxes
                
                if 'labels' in transformed:
                    targets['instances']['labels'] = np.array(transformed['labels'])
            else:
                # No boxes remaining after transform
                targets['instances']['bboxes'] = np.empty((0, 4), dtype=np.float32)
                if labels is not None:
                    targets['instances']['labels'] = np.empty((0,), dtype=labels.dtype)
        else:
            # No bounding boxes, just transform the image
            transformed = self.transform(image=img, bboxes=[], labels=[])
            img = transformed['image']
        
        # Update image dimensions
        targets['height'], targets['width'] = img.shape[0], img.shape[1]
        
        return img, targets
    
class Augmentation(nn.Module):
    '''Collection of image augmentation techniques for object detection.
    
    This class provides a set of common image augmentations that are useful in object detection,
    implemented using the Albumentations library. Augmentations include:
    - Blur effects (Gaussian blur and median blur)
    - Grayscale conversion
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - HSV color space shifts (hue, saturation, value)
    - Horizontal flipping
    
    The augmentations are applied with configurable probabilities and parameters.
    Bounding box coordinates are automatically transformed along with the image.
    
    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords (optional)
        targets['instances']['labels']: (N,) np.int64 (optional)
        
    Output format: Same as input, but with augmented img and updated targets
    
    Args:
        blur_p (float): Probability of applying Gaussian blur. Defaults to 0.01.
        median_blur_p (float): Probability of applying median blur. Defaults to 0.01.
        to_gray_p (float): Probability of converting to grayscale. Defaults to 0.01.
        clahe_p (float): Probability of applying CLAHE. Defaults to 0.01.
        hue_shift_limit (float): Max hue shift as fraction of 180. Defaults to 0.015.
        sat_shift_limit (float): Max saturation shift as fraction of 100. Defaults to 0.7.
        val_shift_limit (float): Max value shift as fraction of 100. Defaults to 0.4.
        horizontal_flip_p (float): Probability of horizontal flipping. Defaults to 0.5.
    '''
    def __init__(self, blur_p=0.01, median_blur_p=0.01, to_gray_p=0.01, clahe_p=0.01, 
                 hue_shift_limit=0.015, sat_shift_limit=0.7, val_shift_limit=0.4, horizontal_flip_p=0.5):
        super().__init__()
        transform = A.Compose([
            A.Blur(p=blur_p),  
            A.MedianBlur(p=median_blur_p),
            A.ToGray(p=to_gray_p),   
            A.CLAHE(p=clahe_p),
            A.HueSaturationValue(
                hue_shift_limit=hue_shift_limit * 180,  # hue_delta=0.015
                sat_shift_limit=sat_shift_limit * 100,    # saturation_delta=0.7
                val_shift_limit=val_shift_limit * 100,    # value_delta=0.4
                p=1.0   
            ),
            A.HorizontalFlip(p=horizontal_flip_p),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
        
        self.transform = AlbumentationsTransform(transform)
        
    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        data = self.transform(data)
        return data


class SanitizeBoundingBoxes(nn.Module):
    """
    Ensures that bounding boxes are valid.

    This transform sanitizes bounding boxes by:
    1. Clipping them to the image boundaries.
    2. Filtering out boxes that are smaller than a minimum size after clipping.
    3. Filtering out boxes that are completely outside the image boundaries.
    4. Filtering out degenerate boxes (where x1 >= x2 or y1 >= y2).
    5. Updating all corresponding instance-level annotations like labels, texts.

    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords
        targets['instances']['labels']: (N,) np.int64 (optional)
        targets['instances']['texts']: list (optional, for multimodal)

    Output format: Same as input, but with sanitized bboxes and corresponding annotations.

    Args:
        min_size (float): The minimum width or height for a bounding box to be kept.
                          Defaults to 1.0.
        min_area_ratio (float): Minimum ratio of clipped area to original area.
                               Defaults to 0.1 (10%).
    """
    def __init__(self, min_size: float = 1.0, min_area_ratio: float = 0.1):
        super().__init__()
        self.min_size = min_size
        self.min_area_ratio = min_area_ratio

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        
        if 'instances' not in targets or 'bboxes' not in targets['instances'] or targets['instances']['bboxes'].shape[0] == 0:
            return data

        h, w = img.shape[:2]
        bboxes = targets['instances']['bboxes']
        num_orig_bboxes = len(bboxes)

        # Store original areas for area ratio check
        orig_w = bboxes[:, 2] - bboxes[:, 0]
        orig_h = bboxes[:, 3] - bboxes[:, 1]
        orig_areas = orig_w * orig_h

        # 1. Check for degenerate boxes (x1 >= x2 or y1 >= y2)
        valid_shape = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

        # 2. Check if boxes have any overlap with image (not completely outside)
        in_bounds = (bboxes[:, 0] < w) & (bboxes[:, 1] < h) & (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)

        # 3. Clip boxes to image boundaries
        clipped_bboxes = bboxes.copy()
        clipped_bboxes[:, [0, 2]] = np.clip(clipped_bboxes[:, [0, 2]], 0, w)
        clipped_bboxes[:, [1, 3]] = np.clip(clipped_bboxes[:, [1, 3]], 0, h)

        # 4. Check size after clipping
        clipped_w = clipped_bboxes[:, 2] - clipped_bboxes[:, 0]
        clipped_h = clipped_bboxes[:, 3] - clipped_bboxes[:, 1]
        size_valid = (clipped_w >= self.min_size) & (clipped_h >= self.min_size)

        # 5. Check area ratio (how much of the original box is still visible)
        clipped_areas = clipped_w * clipped_h
        area_ratio_valid = (orig_areas < 1e-6) | ((clipped_areas / (orig_areas + 1e-6)) >= self.min_area_ratio)

        # Combine all validation criteria
        valid_mask = valid_shape & in_bounds & size_valid & area_ratio_valid
        
        if np.all(valid_mask):
            targets['instances']['bboxes'] = clipped_bboxes
            return data

        valid_indices = np.where(valid_mask)[0]
        
        # Update bounding boxes
        targets['instances']['bboxes'] = clipped_bboxes[valid_indices]
        
        # Update labels if present
        if 'labels' in targets['instances']:
            labels = targets['instances']['labels']
            if isinstance(labels, (np.ndarray, list)) and len(labels) == num_orig_bboxes:
                if isinstance(labels, np.ndarray):
                    targets['instances']['labels'] = labels[valid_indices]
                else:  # list
                    targets['instances']['labels'] = [labels[i] for i in valid_indices]

        return img, targets


class RandomLoadTexts(nn.Module):
    def __init__(self, num_texts: int = 80, blank_text: bool = False):
        super().__init__()
        self.num_texts = num_texts
        self.blank_text = blank_text

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        texts = targets['instances']['texts']
        positive_texts = [texts[i] for i in targets['instances']['labels']]
        positive_texts = list(set(positive_texts))
        if targets.get('detection', False):
            dataset = targets['dataset']
            class_texts = dataset.detect_text_list
            remaining_texts = set(class_texts) - set(positive_texts)
            additional_texts = random.sample(list(remaining_texts), self.num_texts - len(positive_texts))
            positive_texts.extend(additional_texts)
        elif not self.blank_text:
            dataset = targets['dataset']
            class_texts = dataset.global_text_list
            remaining_texts = set(class_texts) - set(positive_texts)
            additional_texts = random.sample(list(remaining_texts), self.num_texts - len(positive_texts))
            positive_texts.extend(additional_texts)
        else:
            positive_texts.extend([' '] * (self.num_texts - len(positive_texts)))
            
        text2id = {text: i for i, text in enumerate(positive_texts)}

        labels = []
        for i, label in enumerate(targets['instances']['labels']):
            lookup_key = texts[label]
            labels.append(text2id[lookup_key])

        targets['instances']['labels'] = np.array(labels)
        targets['instances']['texts'] = positive_texts
        
        return img, targets
            
class ViewAugmentation(nn.Module):
    '''View augmentation class using albumentations library with specific transforms.
    
    This class implements a predefined set of data augmentations using the Albumentations library.
    The augmentations include:
    - RandomHorizontalFlip (p=0.5): Randomly flips the image horizontally
    - ColorJitter (p=0.8): Applies random brightness, contrast, saturation, and hue changes
        - Brightness: 0.4
        - Contrast: 0.4  
        - Saturation: 0.2
        - Hue: 0.1
    - GaussianBlur (p=0.5): Applies Gaussian blur to the image
    - RandomSolarize (p=0.2, threshold=128): Randomly solarizes the image
    
    Input format: img (np.ndarray HWC), targets (dict)
        targets['height'], targets['width']: input image dimensions
        targets['instances']['bboxes']: (N, 4) np.float32, format xyxy absolute coords (optional)
        targets['instances']['labels']: (N,) np.int64 (optional)
        
    Output format: Same as input, but with augmented img and updated targets:
        - targets['width'], targets['height']: updated to new dimensions
        - targets['instances']['bboxes']: transformed bboxes if present
        - targets['instances']['labels']: updated labels if present
    '''
    def __init__(self):
        super().__init__()
        
        # Define the augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=(0.875, 1.125),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=0.0,
                p=0.5
            ),
            A.GaussianBlur(p=0.2),
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
        
        self.transform = AlbumentationsTransform(transform)
        
    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        data = self.transform(data)
        return data


class LoadTexts(nn.Module):
    def __init__(self, text_path: str):
        super().__init__()
        with open(text_path, 'r') as f:
            self.texts = json.load(f)
        self.texts_list = [t[0] for t in self.texts]

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        targets['instances']['texts'] = self.texts_list
        return img, targets
    
    
class LimitBboxes(nn.Module):
    def __init__(self, max_bboxes: int = 300):
        super().__init__()
        self.max_bboxes = max_bboxes
        
    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        bboxes = targets['instances']['bboxes']
        labels = targets['instances']['labels']
        if len(bboxes) > self.max_bboxes:
            indices = np.random.permutation(len(bboxes))[:self.max_bboxes]
            bboxes = bboxes[indices]
            labels = labels[indices]
        targets['instances']['bboxes'] = bboxes
        targets['instances']['labels'] = labels
        return img, targets
   
    
class ToTensor(nn.Module):
    """Convert image/targets to tensors, optionally normalize image."""
    def __init__(
        self,
        normalize: bool = True,
        mean: Optional[list] = [0.485, 0.456, 0.406],
        std: Optional[list] = [0.229, 0.224, 0.225],
        process_targets: bool = True
    ):
        super().__init__()
        self.normalize = normalize
        if normalize:
            assert mean is not None and std is not None, \
                "mean/std must be provided when normalize=True"
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(-1, 1, 1))
        self.process_targets = process_targets

    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        targets.pop("dataset", None)

        # Convert image: HWC (uint8) -> CHW (uint8)
        if img.ndim == 3:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC -> CHW
        elif img.ndim == 2:
            img = np.ascontiguousarray(img[np.newaxis, :, :])  # HW -> CHW (gray)
        img_tensor = torch.from_numpy(img).to(torch.float32) / 255.0  # normalize to [0,1]

        # Optional Normalize (channel-wise)
        if self.normalize:
            img_tensor = (img_tensor - self.mean) / self.std
            
        if not self.process_targets:
            return img_tensor

        # Convert target tensors
        if "instances" in targets:
            for key, value in targets["instances"].items():
                if isinstance(value, np.ndarray):
                    targets["instances"][key] = torch.from_numpy(value).to(torch.float32)
                elif key == "texts" and isinstance(value, list):
                    pass

        # Convert other numpy arrays
        for key, value in targets.items():
            if isinstance(value, np.ndarray):
                targets[key] = torch.from_numpy(value).to(torch.float32)

        return img_tensor, targets
    
class BackImg(nn.Module):
    def __init__(self, size: Tuple[int, int] = (640, 640),):
        super().__init__()
        self.aug = Compose([
            KeepRatioResize(size=size),
            LetterResize(size=size),
            ToTensor(process_targets=False)
                ])
    
    def forward(self, data):
        img, targets = data
        data_cache_len = targets['dataset'].dataset.get_len_cache()
        indices = [random.randint(0, data_cache_len-1)]
        data = targets['dataset'].dataset.get_data_from_cache(indices)[0]
        img_bg = self.aug(data)
        targets['img_bg'] = img_bg
        return img, targets
    
class Mixup(nn.Module):
    def __init__(self, prob: float = 0.5, 
                 size: Tuple[int, int] = (640, 640),  
                 apply_moasic: bool = False,               
                 ):
        super().__init__()
        self.prob = prob
        self.aug_base = Compose([
            KeepRatioResize(size=size),
            LetterResize(size=size),
            RandomAffine(max_rotate_degree=0.0, max_shear_degree=0.0,
                        scaling_ratio_range=(0.5, 1.5), max_aspect_ratio=100.0,
                        border_val=(114, 114, 114)),
                ])
        self.aug_moasic = Compose([
            MultiModalMoasic(size=size, center_ratio_range=(0.5, 1.5), min_bbox_size=1.0),
            RandomAffine(max_rotate_degree=0.0, max_translate_ratio=0.1, 
                        scaling_ratio_range=(0.5, 1.5), max_shear_degree=0.0, 
                        border=(-320, -320), border_val=(114, 114, 114), bbox_clip_border=True),
            Augmentation()
        ])
        self.apply_moasic = apply_moasic
        
    def _get_mixup_data(self, mixup_dataset):
        for _ in range(10):
            try:
                if not self.apply_moasic:
                    if len(mixup_dataset) == 3:
                        a = [0, 1, 2]
                        i = np.random.choice(a, p=[0.45, 0.45, 0.1], size=1)
                        mixup_dataset = mixup_dataset[i.item()]
                    else:
                        mixup_dataset = mixup_dataset[0]
                    data_cache_len = mixup_dataset.get_len_cache()
                    image, targets = mixup_dataset.get_data_from_cache(random.randint(0, data_cache_len-1))
                    image, targets = self.aug_base((image, targets))
                    break
                else:
                    data = []
                    for i in range(4):
                        if len(mixup_dataset) == 3:
                            a = [0, 1, 2]
                            i = np.random.choice(a, p=[0.45, 0.45, 0.1], size=1)
                            mixup_dataset = mixup_dataset[i.item()]
                        else:
                            mixup_dataset = mixup_dataset[0]
                        data_cache_len = mixup_dataset.get_len_cache()
                        image, targets = mixup_dataset.get_data_from_cache(random.randint(0, data_cache_len-1))
                        data.append((image, targets))
                    image, targets = self.aug_moasic(data)
                    break
            except Exception as e:
                print(e)
                continue
        return image, targets
        
    def forward(self, data: Tuple[np.ndarray, Dict[str, Any]]):
        img, targets = data
        if random.random() >= self.prob:
            targets['mixup'] = False
            return data
        
        mixup_dataset = targets['dataset'].mixup_dataset
        mixup_img, mixup_targets = self._get_mixup_data(mixup_dataset)

        beta = round(random.uniform(0.45, 0.55), 6)
        img = img * beta + mixup_img * (1 - beta)

        combined_instances = [targets['instances'], mixup_targets['instances']]

        texts = []
        for instance in combined_instances:
            for label in instance['labels']:
                texts.append(instance['texts'][label])
        texts = list(set(texts))  

        text2id = {text: i for i, text in enumerate(texts)}
        labels = []
        bboxes = []
        for instance in combined_instances:
            for i, label in enumerate(instance['labels']):
                lookup_key = instance['texts'][label]
                labels.append(text2id[lookup_key])
            bboxes.append(instance['bboxes'])

        targets['instances']['labels'] = np.array(labels)
        targets['instances']['bboxes'] = np.concatenate(bboxes, axis=0)
        targets['instances']['texts'] = texts
        targets['mixup'] = True
        return img, targets


class MultiModalMoasic(nn.Module):
    '''This code is a modified version of the MultiModalMosaic in YOLO-World:
    https://github.com/AILab-CVC/YOLO-World/blob/master/yolo_world/datasets/transformers/mm_mix_img_transforms.py

    '''
    def __init__(self,
                 size: Tuple[int, int] = (640, 640),
                 prob: float = 1.0,
                 pad_val: int = 114,
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 min_bbox_size = 1.0,
                 ):
        super().__init__()
        self.size = size
        self.prob = prob
        self.pad_val = pad_val
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size

    def _update_texts(self, mix_data: List[Tuple[np.ndarray, Dict[str, Any]]]):
        """Combines unique texts from all data samples and creates a mapping."""
        texts = [text for data in mix_data for text in data[1]['instances']['texts']]
        texts = list(set(texts))
        text2id = {text: i for i, text in enumerate(texts)}
        for i, data in enumerate(mix_data):
            texts_list = data[1]['instances']['texts']
            for j, label in enumerate(data[1]['instances']['labels']):
                text = texts_list[label]
                updated_id = text2id[text]
                data[1]['instances']['labels'][j] = updated_id
        return texts, mix_data
    
    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.size[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.size[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.size[0] * 2), \
                             min(self.size[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def forward(self, data: Union[Tuple[np.ndarray, Dict[str, Any]], List[Tuple[np.ndarray, Dict[str, Any]]]]):
        if random.random() > self.prob:
            return data
        
        if isinstance(data, tuple):
            img, targets = data
            data_cache_len = targets['dataset'].dataset.get_len_cache()
            indices = [random.randint(0, data_cache_len-1) for _ in range(3)]
            mosaic_data = [data] + targets['dataset'].dataset.get_data_from_cache(indices)
        else:
            mosaic_data = data
            
        texts, mosaic_data = self._update_texts(mosaic_data)

        mosaic_img = np.full((int(self.size[0] * 2), int(self.size[1] * 2), 3),
                            self.pad_val, dtype=img.dtype)
        
        center_position = (
            int(np.rint(random.uniform(*self.center_ratio_range) * self.size[0])),
            int(np.rint(random.uniform(*self.center_ratio_range) * self.size[1]))
        )

        mosaic_bboxes, mosaic_labels, mosaic_ignore = [], [], []
        
        for i, loc in enumerate(('top_left', 'top_right', 'bottom_left', 'bottom_right')):
            img_i, targets_i = mosaic_data[i]
            scale_ratio = min(self.size[1] / img_i.shape[0], self.size[0] / img_i.shape[1])
            new_w = int(np.rint(img_i.shape[1] * scale_ratio))
            new_h = int(np.rint(img_i.shape[0] * scale_ratio))
            img_i = cv2.resize(img_i, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            
            # Ensure coordinates are within bounds
            paste_x1, paste_y1, paste_x2, paste_y2 = paste_coord
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coord
            
            # Clip paste coordinates to mosaic image bounds
            paste_x1 = max(0, min(mosaic_img.shape[1], paste_x1))
            paste_y1 = max(0, min(mosaic_img.shape[0], paste_y1))
            paste_x2 = max(0, min(mosaic_img.shape[1], paste_x2))
            paste_y2 = max(0, min(mosaic_img.shape[0], paste_y2))
            
            # Clip crop coordinates to image bounds
            crop_x1 = max(0, min(img_i.shape[1], crop_x1))
            crop_y1 = max(0, min(img_i.shape[0], crop_y1))
            crop_x2 = max(0, min(img_i.shape[1], crop_x2))
            crop_y2 = max(0, min(img_i.shape[0], crop_y2))
            
            # Check if there's valid area to paste
            if paste_x2 > paste_x1 and paste_y2 > paste_y1 and crop_x2 > crop_x1 and crop_y2 > crop_y1:
                # Adjust crop size to match paste size if needed
                paste_w, paste_h = paste_x2 - paste_x1, paste_y2 - paste_y1
                crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
                
                if paste_w != crop_w or paste_h != crop_h:
                    # Resize crop to match paste area
                    crop_img = img_i[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_img.size > 0:
                        crop_img = cv2.resize(crop_img, (paste_w, paste_h), interpolation=cv2.INTER_LINEAR)
                        mosaic_img[paste_y1:paste_y2, paste_x1:paste_x2] = crop_img
                else:
                    mosaic_img[paste_y1:paste_y2, paste_x1:paste_x2] = img_i[crop_y1:crop_y2, crop_x1:crop_x2]

            # Transform bboxes with better precision
            if 'bboxes' in targets_i['instances'] and targets_i['instances']['bboxes'].shape[0] > 0:
                bboxes = targets_i['instances']['bboxes'].copy().astype(np.float32)
                bboxes = bboxes * scale_ratio
                bboxes[:, [0, 2]] += paste_coord[0] - crop_coord[0]
                bboxes[:, [1, 3]] += paste_coord[1] - crop_coord[1]
                
                # Clip bboxes to mosaic image bounds
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, mosaic_img.shape[1])
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, mosaic_img.shape[0])
            else:
                bboxes = np.empty((0, 4), dtype=np.float32)
            
            mosaic_bboxes.append(bboxes)
            
            # Handle labels and ignore_flags safely
            if 'labels' in targets_i['instances'] and len(targets_i['instances']['labels']) > 0:
                labels = targets_i['instances']['labels']                
                mosaic_labels.append(labels)
            else:
                mosaic_labels.append(np.empty((0,), dtype=np.int64))
                
            mosaic_ignore.append(np.zeros(len(bboxes), dtype=bool))

        mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
        mosaic_labels = np.concatenate(mosaic_labels, axis=0)

        # Filter small boxes
        valid_mask = ((mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0]) >= self.min_bbox_size) & \
                    ((mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1]) >= self.min_bbox_size)

        return mosaic_img, {
            'height': self.size[1],
            'width': self.size[0],
            'dataset': targets['dataset'],
            'instances': {
                'bboxes': mosaic_bboxes[valid_mask],
                'labels': mosaic_labels[valid_mask],
                'texts': texts
            },
            'detection': targets['detection']
        }
