"""
Minimal Multi-Query Multi-Head Attention with PyTorch Flash Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MQAttention(nn.Module):
    """Multi-Query Attention: n_head query heads, n_kv_head key-value heads."""

    def __init__(self, dim, n_head=8, n_kv_head=1):
        super().__init__()
        assert dim % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * n_kv_head, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * n_kv_head, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Repeat kv heads to match query heads: (B, n_kv, L, d) -> (B, n_head, L, d)
        rep = self.n_head // self.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # PyTorch flash attention (auto-dispatches to FlashAttention / memory-efficient kernel)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)

        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))
