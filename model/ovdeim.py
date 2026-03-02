"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'


class GatedFFNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * dim * 2) 
        self.w3 = nn.Linear(dim * 2, dim)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x1 = self.norm(x)
        x12 = self.w12(x1)
        x1, x2 = x12.chunk(2, dim=-1)
        gated = F.silu(x1) * x2
        out = x + self.w3(gated)
        return out


class TextAdapter(nn.Module):
    def __init__(self, text_dim: int, img_dim: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedFFNBlock(text_dim) for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(text_dim, img_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.layers:
            init.xavier_uniform_(layer.w12.weight)
            init.constant_(layer.w12.bias, 0)
            init.xavier_uniform_(layer.w3.weight)
            init.constant_(layer.w3.bias, 0)
        init.xavier_uniform_(self.proj_out.weight)
        init.constant_(self.proj_out.bias, 0)

    def forward(self, text_feats):
        x = text_feats
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)


class OVDEIM(nn.Module):
    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
        img_dim: int,
        text_dim:int,
        text_layers: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.text_adapter = TextAdapter(text_dim, img_dim, text_layers)
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        text_feats = self.text_adapter(targets['text_feats'] ) 
        x = self.encoder(x)        
        x = self.decoder(x, targets, text_feats)
        return x
