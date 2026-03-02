import os.path
from typing import Any, Tuple, Union, Dict
import torch
import json
import numpy as np
from torch.utils.data import Dataset
import random
import pickle
import torch.multiprocessing as mp
import bisect

from collections import deque
import copy

from pycocotools.coco import COCO
from pathlib import Path
import cv2
import gc

class MultiModalDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        detection: bool = True,
        lvis: bool = False,
        lvis_path: str = None,
        return_img: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.detection = detection
        self.lvis = lvis
        self.coco = COCO(annFile)
        self.ids = self.coco.getImgIds()
        self.return_img = return_img
        if self.detection:
            cats = self.coco.cats 
            sorted_cats = {i: cats[i] for i in sorted(cats)} 
            self.coco.cats = sorted_cats
            if not self.lvis:  
                self.class_texts = [v['name'].split("/")[0] for k, v in sorted_cats.items()]
            else: 
                with open(lvis_path, 'r') as f:
                    texts = json.load(f)
                self.class_texts = [t[0] for t in texts]
                
    def _load_image_path(self, id: int):
        if self.lvis:
            path = self.coco.loadImgs([id])[0]['coco_url'].replace(
                'http://images.cocodataset.org/', '')
        else:
            path = self.coco.loadImgs([id])[0]["file_name"]
        return os.path.join(self.root, path)

    def _load_targets_detection(self, id: int):
        img_info = self.coco.loadImgs([id])[0]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(id))
        targets = {}
        targets['height'] = int(img_info['height'])
        targets['width'] = int(img_info['width'])
        bboxes, labels, texts = [], [], []        
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False) or ann.get('iscrowd', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            category_id = ann['category_id'] - 1
            if category_id < 0 or category_id >= len(self.class_texts):
                print(f"Warning: category_id {ann['category_id']} is out of range for class_texts with length {len(self.class_texts)}")
                continue

            bboxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(category_id)
            texts.append(self.class_texts[category_id])
        
        if not self.lvis:
            texts_unique = list(set(texts))
            cat2id = {text: i for i, text in enumerate(texts_unique)}
            labels = [cat2id[text] for text in texts]
        
        targets['instances'] = {
            'img_id': id,
            'ori_shape': np.array([targets['height'], targets['width']], dtype=np.int64),
            'bboxes': np.array(bboxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'texts': texts_unique if not self.lvis else self.class_texts,
        }
        if len(targets['instances']['texts']) == 0:
            return None
        targets['detection'] = True
        return targets
    
    def _load_targets_grounding(self, id: int):
        img_info = self.coco.loadImgs([id])[0]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(id))
        targets = {}
        targets['height'] = int(img_info['height'])
        targets['width'] = int(img_info['width'])
        bboxes, labels, texts = [], [], []
        cat2id = {} 
        for ann in ann_info:
            cat_name = ' '.join([img_info['caption'][t[0]:t[1]]
                                 for t in ann['tokens_positive']]).lower().strip()
            if cat_name not in cat2id:
                cat2id[cat_name] = len(cat2id)
                texts.append(cat_name)

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False) or ann.get('iscrowd', False):
                continue
            x1, y1, w, h = ann['bbox']  
            inter_w = max(0,
                          min(x1 + w, float(img_info['width'])) - max(x1, 0))
            inter_h = max(0,
                          min(y1 + h, float(img_info['height'])) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if float(ann['area']) <= 0 or w < 1 or h < 1:
                continue

            bboxes.append([x1, y1, x1 + w, y1 + h])
            cat_name = ' '.join([img_info['caption'][t[0]:t[1]]
                                for t in ann['tokens_positive']]).lower().strip()
            labels.append(cat2id[cat_name])
        targets['instances'] = {
            'img_id': id,
            'ori_shape': np.array([targets['height'], targets['width']], dtype=np.int64),
            'bboxes': np.array(bboxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'texts': texts,
        }
        targets['detection'] = False
        if len(targets['instances']['texts']) == 0:
            return None
        return targets
    
    def _load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index >= len(self.ids):
            raise IndexError(f"Index {index} is out of range for ids list with length {len(self.ids)}")
        
        id = self.ids[index]
        image_path = self._load_image_path(id)
        if self.detection:
            targets = self._load_targets_detection(id)
        else:
            targets = self._load_targets_grounding(id)  
        if self.return_img:
            image = self._load_image(image_path)
            return image, targets
        else:
            return image_path, targets

    def __len__(self) -> int:
        return len(self.ids)
    
class CacheMultiModalDataset(Dataset):
    def __init__(self, dataset, cache_size: int = 500, data_ratio: float = 1.0, use_cache: bool = False):
        self.dataset = dataset
        self.data_ratio = data_ratio
        self.data_bytes, self.data_address = self._serialize_data()
        
        self.cache_size = cache_size
        self.use_cache = use_cache
        
        if self.use_cache:
            self.data_cache = deque(maxlen=self.cache_size)
            self.cache_lock = mp.Lock()
            self._prefill_cache()
        else:
            self.data_cache = None
            self.cache_lock = None
        
    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)
        
        valid_indices = []
        for i in range(len(self.dataset)):
            try:
                data = self.dataset[i]
                if data[1] is not None:
                    valid_indices.append(i)
            except Exception as e:
                print(f"Error processing data at index {i}: {e}")
                continue
        data_ratio = self.data_ratio  
        num_samples = max(1, int(len(valid_indices) * data_ratio))
        selected_indices = random.sample(valid_indices, num_samples)
        print(f"Total valid samples: {len(valid_indices)}, Selected samples: {num_samples}")
        
        serialized_data_list = []
        for i in selected_indices:
            try:
                data = self.dataset[i]
                serialized_data_list.append(_serialize(data))
            except Exception as e:
                print(f"Error processing data at index {i}: {e}")
                continue
        
        address_list = np.asarray([len(x) for x in serialized_data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        data_bytes = np.concatenate(serialized_data_list)
        
        # Clear list to free memory
        serialized_data_list.clear()
        self.dataset = None
        self.dataset = self
        gc.collect()
        
        return data_bytes, data_address
    
    def __len__(self):
        return len(self.data_address)
    
    def _load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_data_serialized(self, idx):
        for _ in range(10):
            try:
                if idx >= len(self.data_address):
                    raise IndexError(f"Index {idx} is out of range for data_address with length {len(self.data_address)}")
                
                start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
                end_addr = self.data_address[idx].item()
                mv = memoryview(
                    self.data_bytes[start_addr:end_addr]) 
                image_path, targets = pickle.loads(mv) 
                image = self._load_image(image_path)
            except Exception as e:
                print(f"Failed to load targets at index {idx}: {e}")
                targets = None
            if targets == None:
                idx = random.randint(0, self.__len__() -1)
            else:
                break
        return image, targets
    
    def _prefill_cache(self):
        if not self.use_cache:
            return
        n_prefill = min(self.cache_size, self.__len__())
        for idx in range(n_prefill):
            img, targets = self.get_data_serialized(idx)
            self.data_cache.append((copy.deepcopy(img), copy.deepcopy(targets)))
            
    def update_cache(self, img, targets):
        if not self.use_cache:
            return
        with self.cache_lock:
            self.data_cache.append((copy.deepcopy(img), copy.deepcopy(targets)))
            
    def get_len_cache(self):
        if self.use_cache:
            return len(self.data_cache)
        else:
            return self.__len__()
    
    def get_data_from_cache(self, indices):
        if self.use_cache:
            if isinstance(indices, int):
                indices = [indices]
                return_item = True
            else:
                return_item = False
            data = []
            with self.cache_lock:
                for i in indices:
                    if 0 <= i < len(self.data_cache):
                        img, targets = self.data_cache[i]
                        data.append((copy.deepcopy(img), copy.deepcopy(targets)))
                    else:
                        raise IndexError(f"Cache index {i} out of range (len={len(self.data_cache)})")
            if return_item:
                return data[0]
            else:
                return data
        else:
            if isinstance(indices, int):
                return self.get_data_serialized(indices)
            else:
                data = []
                for idx in indices:
                    data.append(self.get_data_serialized(idx))
                return data
        
    def __getitem__(self, index):        
        img, targets = self.get_data_serialized(index)
        if self.use_cache:
            self.update_cache(img, targets)
        return img, targets

class MultiModalDatasetAugmented(Dataset):
    def __init__(self, dataset, cache_dir, 
                 detect_dir_text=None, global_dir_text=None, 
                 pipeline_aug=None, pipeline_clean=None, max_retries=5, test=False,
                 prob_aug = 1.0):
        self.dataset = dataset
        self.text_list, self.text_to_index, self.embeddings = self.load_text_embeddings(cache_dir)
        self.pipeline_aug = pipeline_aug
        self.pipeline_clean = pipeline_clean
        self.max_retries = max_retries
        self.test = test
        self.prob_aug = prob_aug
        if detect_dir_text is not None:
            self.detect_text_list = self._load_text_list(detect_dir_text)
        if global_dir_text is not None:
            self.global_text_list = self._load_text_list(global_dir_text)
            
        print(
            "=== Pipeline Configuration ===\n"
            f"  • pipeline_aug:   {pipeline_aug}\n"
            f"  • pipeline_clean: {pipeline_clean}\n"
            f"  • prob_aug:       {prob_aug}\n"
            "=============================="
        )

    def _load_text_list(self, cache_dir_text):
        with open(cache_dir_text, 'r') as f:
            text_list = json.load(f)
        return text_list
    
    def load_text_embeddings(self, cache_dir):
        cache_data = torch.load(cache_dir)
        text_list = cache_data['text_list']
        text_to_index = cache_data['text_to_index']
        embeddings = cache_data['embeddings']
        return text_list, text_to_index, embeddings
    
    def get_data(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        for _ in range(self.max_retries):
            try:
                image, targets = self.get_data(idx)
            except IndexError as e:
                print(f"IndexError in get_data: {e}")
                idx = random.randint(0, len(self)-1)
                continue
            if image is not None:
                try:
                    targets['dataset'] = self
                    if self.pipeline_clean is not None or self.pipeline_aug is not None:
                        if random.random() < self.prob_aug and not self.test:
                            image, targets = self.pipeline_aug((image, targets))
                            targets['aug'] = True
                        else:
                            image, targets = self.pipeline_clean((image, targets))
                            targets['aug'] = False
                        texts = targets['instances']['texts']
                        if not self.test:
                            text_ids = []
                            for text in texts:
                                if text in self.text_to_index:
                                    text_id = self.text_to_index[text]
                                    if text_id < len(self.embeddings):
                                        text_ids.append(text_id)
                                    else:
                                        print(f"Warning: text_id {text_id} for text '{text}' is out of range for embeddings with length {len(self.embeddings)}")
                                else:
                                    raise ValueError(f"text '{text}' not found in text_to_index")
                            
                            targets['text_ids'] = text_ids
                            targets['text_feats'] = self.embeddings[text_ids]
                        else:
                            text_ids = []
                            for text in texts:
                                if text in self.text_to_index:
                                    text_id = self.text_to_index[text]
                                    if text_id < len(self.embeddings):
                                        text_ids.append(text_id)
                                    else:
                                        print(f"Warning: text_id {text_id} for text '{text}' is out of range for embeddings with length {len(self.embeddings)}")
                                else:
                                    raise ValueError(f"text '{text}' not found in text_to_index")
                            
                            targets['text_ids'] = text_ids
                            targets['text_feats'] = self.embeddings[text_ids]
                    return (image, targets)
                except Exception as e:
                    idx = random.randint(0, len(self)-1)
                    print(f"Error processing data at index {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                idx = random.randint(0, len(self)-1)
                continue
        raise RuntimeError("Failed to load data after maximum retries")
            
class ConcatDataset(Dataset):
    def __init__(self, datasets, cache_dir, 
                 detect_dir_text=None, global_dir_text=None, 
                 pipeline_aug=None, pipeline_clean=None, max_retries=15, test=False,
                 epoch=0, prob_aug = 1.0):
        self.datasets = [MultiModalDatasetAugmented(dataset, cache_dir, detect_dir_text, 
                                                    global_dir_text, pipeline_aug, 
                                                    pipeline_clean, max_retries, 
                                                    test, prob_aug) for dataset in datasets]
        self.cumulative_sizes = self._cumsum([len(d) for d in self.datasets])
        self.epoch = epoch
        if len(self.datasets) == 3:
            mixup_dataset = [self.datasets[0].dataset, self.datasets[1].dataset, self.datasets[2].dataset]
        else:
            mixup_dataset = [self.datasets[0].dataset]
            
        for i in range(len(self.datasets)):
            self.datasets[i].mixup_dataset = mixup_dataset    
            
        self.text_list, self.text_to_index, self.embeddings = self.load_text_embeddings(cache_dir)  
        
        if global_dir_text is not None:
            global_text_list = self._load_text_list(global_dir_text)
            self.global_text_ids = [self.text_to_index[text] for text in global_text_list]      
        
        print(f"ConcatDataset initialized:")
        print(f"  - Number of sub-datasets: {len(self.datasets)}")
        print(f"  - Individual dataset sizes: {[len(d) for d in self.datasets]}")
        print(f"  - Cumulative sizes: {self.cumulative_sizes}")
        print(f"  - Total dataset size: {len(self)}")
        
    def _load_text_list(self, cache_dir_text):
        with open(cache_dir_text, 'r') as f:
            text_list = json.load(f)
        return text_list
    
    def load_text_embeddings(self, cache_dir):
        cache_data = torch.load(cache_dir)
        text_list = cache_data['text_list']
        text_to_index = cache_data['text_to_index']
        embeddings = cache_data['embeddings']
        return text_list, text_to_index, embeddings
    
    def _cumsum(self, lengths):
        return [sum(lengths[:i+1]) for i in range(len(lengths))]
    
    def __len__(self):
        return self.cumulative_sizes[-1]
      
    def __getitem__(self, idx):
        try:
            if idx >= len(self):
                raise IndexError(f"Index {idx} is out of range for dataset with length {len(self)}")
                
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            
            if dataset_idx >= len(self.datasets):
                raise IndexError(f"Dataset index {dataset_idx} is out of range for {len(self.datasets)} datasets")
                
            sub_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
            
            if sub_idx >= len(self.datasets[dataset_idx]):
                raise IndexError(f"Sub-index {sub_idx} is out of range for dataset {dataset_idx} with length {len(self.datasets[dataset_idx])}")
                
            img, targets = self.datasets[dataset_idx][sub_idx]
            targets['dataset'] = self
            return img, targets
        except Exception as e:
            print(f"Error in ConcatDataset.__getitem__ with idx {idx}: {e}")
            print(f"Dataset info: total_len={len(self)}, cumulative_sizes={self.cumulative_sizes}")
            print(f"dataset_idx={dataset_idx if 'dataset_idx' in locals() else 'undefined'}")
            print(f"sub_idx={sub_idx if 'sub_idx' in locals() else 'undefined'}")
            raise


class CocoDataset(Dataset):
    def __init__(self, root: Union[str, Path], annFile: str):
        super().__init__()
        self.root = root
        self.coco = COCO(annFile)
        self.ids = self.coco.getImgIds()
        self.catid2label, self.label2catid, self.label_texts = self._get_label_map()
        

    def _get_label_map(self) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda x: x['id'])

        catid2label, label2catid, label_texts = {}, {}, []

        for i, cat in enumerate(cats_sorted):
            catid = cat['id']
            cat_name = cat['name']

            catid2label[catid] = i
            label2catid[i] = catid
            label_texts.append(cat_name)
        return catid2label, label2catid, label_texts    

    def _load_image_path(self, id: int):
        path = self.coco.loadImgs([id])[0]["file_name"]
        return os.path.join(self.root, path)
    
    def _load_targets(self, id: int):
        img_info = self.coco.loadImgs([id])[0]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(id))
        targets = {}
        targets['height'] = int(img_info['height'])
        targets['width'] = int(img_info['width'])
        bboxes, labels = [], []        
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False) or ann.get('iscrowd', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            bboxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(self.catid2label[ann['category_id']])
        targets['instances'] = {
            'img_id': id,
            'ori_shape': np.array([targets['height'], targets['width']], dtype=np.int64),
            'bboxes': np.array(bboxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64),
            'texts': self.label_texts,
        }
        targets['detection'] = True
        if len(targets['instances']['bboxes']) == 0:
            return None
        return targets
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image_path = self._load_image_path(id)
        targets = self._load_targets(id)
        return image_path, targets
