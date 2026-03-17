"""
Minimal Qwen-like LLM.
Key features: RMSNorm, RoPE, GQA (grouped-query attention), SwiGLU FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class QwenConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_head: int = 8
    n_kv_head: int = 2        # GQA
    ffn_hidden: int = 1536     # ~3x dim for SwiGLU
    max_len: int = 2048
    norm_eps: float = 1e-6


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


# ── Rotary Position Embedding ────────────────────────────────────────────────

def precompute_rope(dim, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    angles = torch.outer(t, freqs)  # (max_len, dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    # x: (B, n_head, L, head_dim)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos, sin = cos[:x.shape[2]], sin[:x.shape[2]]  # trim to seq len
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, d2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── GQA with RoPE ────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.dim // cfg.n_head

        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=True)
        self.wk = nn.Linear(cfg.dim, self.head_dim * cfg.n_kv_head, bias=True)
        self.wv = nn.Linear(cfg.dim, self.head_dim * cfg.n_kv_head, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        rep = self.n_head // self.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.up   = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.down  = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ── Transformer Block ────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FFN(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class Qwen(nn.Module):
    def __init__(self, cfg: QwenConfig = QwenConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE
        cos, sin = precompute_rope(cfg.dim // cfg.n_head, cfg.max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx
