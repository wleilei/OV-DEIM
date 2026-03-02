import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
import math

class Warmup(object):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg['lr'] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        self.step()

    def step(self):
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            # self.lr_scheduler.step()
            return
        factor = min(1.0, (self.last_step + 1) / self.warmup_duration)
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            pg['lr'] = factor * self.warmup_end_values[i]
        if self.last_step + 1 == self.warmup_duration:
            self.lr_scheduler.step()  

    def finished(self):
        return self.last_step >= self.warmup_duration


class WarmupConstantCosine(object):
    def __init__(self, optimizer, warmup_steps: int, constant_steps: int, cosine_steps: int, 
                 max_lr: float, min_lr: float = 0.0, last_step: int = -1) -> None:
        """
        Implements a three-phase learning rate schedule: warmup -> constant -> cosine decay
        
        Args:
            optimizer: The optimizer to adjust learning rate for
            warmup_steps: Number of steps for warmup phase
            constant_steps: Number of steps for constant phase  
            cosine_steps: Number of steps for cosine decay phase
            max_lr: Maximum learning rate for the main parameter group (used as reference)
            min_lr: Minimum learning rate ratio relative to max_lr
            last_step: Last step number for resuming training
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.cosine_steps = cosine_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_step = last_step
        
        # Store initial lr for each param group and calculate their ratios relative to max_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.lr_ratios = [base_lr / max_lr for base_lr in self.base_lrs]
        
        # Total steps
        self.total_steps = warmup_steps + constant_steps + cosine_steps
        
        self.step()
    
    def get_lr(self):
        """Calculate learning rate for current step"""
        current_step = self.last_step + 1
        
        if current_step <= self.warmup_steps:
            # Warmup phase: linear increase from 0 to max_lr (maintaining ratios)
            factor = current_step / self.warmup_steps
            return [factor * self.max_lr * ratio for ratio in self.lr_ratios]
        
        elif current_step <= self.warmup_steps + self.constant_steps:
            # Constant phase: maintain max_lr (maintaining ratios)
            return [self.max_lr * ratio for ratio in self.lr_ratios]
        
        elif current_step <= self.total_steps:
            # Cosine decay phase (maintaining ratios)
            cosine_step = current_step - self.warmup_steps - self.constant_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_step / self.cosine_steps))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
            return [lr * ratio for ratio in self.lr_ratios]
        
        else:
            # After all phases, maintain min_lr (maintaining ratios)
            return [self.min_lr * ratio for ratio in self.lr_ratios]
    
    def step(self):
        """Update learning rate"""
        self.last_step += 1
        lrs = self.get_lr()
        
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = lrs[i]
    
    def finished(self):
        """Check if all phases are finished"""
        return self.last_step >= self.total_steps
    
    def get_phase(self):
        """Get current phase name"""
        current_step = self.last_step + 1
        
        if current_step <= self.warmup_steps:
            return "warmup"
        elif current_step <= self.warmup_steps + self.constant_steps:
            return "constant"
        elif current_step <= self.total_steps:
            return "cosine"
        else:
            return "finished"

