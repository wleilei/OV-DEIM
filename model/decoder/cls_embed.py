import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class ClassEmbed(nn.Module):
    def __init__(
        self,
        init_bias: float = 100.0,
        init_scale: float = 15.0,
    ):
        super().__init__()
        bias_value = -math.log(init_bias)
        self.lang_bias = nn.Parameter(torch.full((), bias_value))
        self.lang_scale = nn.Parameter(torch.tensor(init_scale).log())
        
    def forward(self, image_embeds, lang_embeds, mask=None):
        image_norm = F.normalize(image_embeds, p=2, dim=-1)
        lang_norm = F.normalize(lang_embeds, p=2, dim=-1)
        
        logits = image_norm @ lang_norm.transpose(2, 1)  # [batch_size, num_queries, num_classes]
                
        logits = logits * torch.exp(self.lang_scale) + self.lang_bias

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float('-inf'))
        return logits