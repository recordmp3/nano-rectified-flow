"""Minimal Multi-Head Attention supporting MHA / MQA / GQA."""

import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self, dim, n_head=8, n_kv_head=None):
        super().__init__()
        self.n_head = n_head
        self.n_kv = n_kv_head or n_head          # MHA: n_kv=n_head, MQA: n_kv=1, GQA: 1<n_kv<n_head
        self.hd = dim // n_head
        self.wq = nn.Linear(dim, n_head * self.hd, bias=False)
        self.wk = nn.Linear(dim, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(dim, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)

        # GQA expand
        if self.n_kv < self.n_head:
            r = self.n_head // self.n_kv
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        # Attention
        att = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        if mask is None:
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        att = (att + mask).softmax(dim=-1)
        out = att @ v

        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))
