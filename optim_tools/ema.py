"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import torch
import torch.nn as nn
from copy import deepcopy
import math

class ModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000):
        super().__init__()
        self.module = deepcopy(model).eval()
        self.decay = decay 
        self.warmups = warmups
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))
        
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = model.state_dict()            
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self):
        return dict(module=self.module.state_dict(), updates=self.updates)
    
    def load_state_dict(self, state: dict, strict: bool = True):
        self.module.load_state_dict(state['module'], strict=strict)
        if 'updates' in state:
            self.updates = state['updates']

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'