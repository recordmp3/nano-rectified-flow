"""
Minimal Qwen-like LLM using Megatron-Core (TP + CP + PP) + PyTorch FSDP (DP).

Requires: pip install megatron-core
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass

from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


@dataclass
class QwenConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_head: int = 8
    n_kv_head: int = 2
    ffn_hidden: int = 1536
    max_len: int = 2048
    norm_eps: float = 1e-6


# ── RMSNorm ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


# ── RoPE ────────────────────────────────────────────────────────────────────

def precompute_rope(dim, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(torch.arange(max_len).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# ── CP all-gather (autograd-compatible) ─────────────────────────────────────

class _CPGather(torch.autograd.Function):
    """Fwd: all-gather KV along seq dim (dim=2). Bwd: chunk back."""
    @staticmethod
    def forward(ctx, x):
        cp = mpu.get_context_parallel_world_size()
        ctx.cp, ctx.rank = cp, mpu.get_context_parallel_rank()
        parts = [torch.empty_like(x) for _ in range(cp)]
        dist.all_gather(parts, x.contiguous(), group=mpu.get_context_parallel_group())
        return torch.cat(parts, dim=2)

    @staticmethod
    def backward(ctx, grad):
        return grad.chunk(ctx.cp, dim=2)[ctx.rank].contiguous()


# ── Attention (Megatron TP + CP + RoPE + GQA) ──────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        tp = mpu.get_tensor_model_parallel_world_size()
        self.n_head = cfg.n_head // tp
        self.n_kv = cfg.n_kv_head // tp
        self.hd = cfg.dim // cfg.n_head

        # Megatron handles column/row sharding + TP comm internally
        self.wq = ColumnParallelLinear(cfg.dim, cfg.dim, bias=True, gather_output=False)
        self.wk = ColumnParallelLinear(cfg.dim, cfg.n_kv_head * self.hd, bias=True, gather_output=False)
        self.wv = ColumnParallelLinear(cfg.dim, cfg.n_kv_head * self.hd, bias=False, gather_output=False)
        self.wo = RowParallelLinear(cfg.dim, cfg.dim, bias=False, input_is_parallel=True)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q, _ = self.wq(x)
        k, _ = self.wk(x)
        v, _ = self.wv(x)
        q = q.view(B, L, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, L, self.n_kv, self.hd).transpose(1, 2)
        v = v.view(B, L, self.n_kv, self.hd).transpose(1, 2)

        # RoPE with CP offset
        cp = mpu.get_context_parallel_world_size()
        off = mpu.get_context_parallel_rank() * L
        q = apply_rope(q, cos[off:off + L], sin[off:off + L])
        k = apply_rope(k, cos[off:off + L], sin[off:off + L])

        # CP: all-gather K,V so every rank sees the full context
        if cp > 1:
            k, v = _CPGather.apply(k), _CPGather.apply(v)

        # GQA expand
        rep = self.n_head // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Flash attention
        if cp <= 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            full_L = L * cp
            qi = torch.arange(off, off + L, device=x.device)
            ki = torch.arange(full_L, device=x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=qi[:, None] >= ki[None, :])

        out, _ = self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))
        return out


# ── SwiGLU FFN (Megatron TP) ────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.gate = ColumnParallelLinear(cfg.dim, cfg.ffn_hidden, bias=False, gather_output=False)
        self.up = ColumnParallelLinear(cfg.dim, cfg.ffn_hidden, bias=False, gather_output=False)
        self.down = RowParallelLinear(cfg.ffn_hidden, cfg.dim, bias=False, input_is_parallel=True)

    def forward(self, x):
        g, _ = self.gate(x)
        u, _ = self.up(x)
        out, _ = self.down(F.silu(g) * u)
        return out


# ── Transformer Block ───────────────────────────────────────────────────────

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


# ── Qwen Model (PP-aware) ───────────────────────────────────────────────────

class Qwen(nn.Module):
    def __init__(self, cfg: QwenConfig = QwenConfig()):
        super().__init__()
        self.cfg = cfg
        pp = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()

        # PP: each stage owns a contiguous slice of layers
        per = (cfg.n_layers + pp - 1) // pp
        l0, l1 = pp_rank * per, min(pp_rank * per + per, cfg.n_layers)

        self.tok_emb = VocabParallelEmbedding(cfg.vocab_size, cfg.dim) if mpu.is_pipeline_first_stage() else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(l1 - l0)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps) if mpu.is_pipeline_last_stage() else None
        self.lm_head = ColumnParallelLinear(cfg.dim, cfg.vocab_size, bias=False) if mpu.is_pipeline_last_stage() else None

        cos, sin = precompute_rope(cfg.dim // cfg.n_head, cfg.max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, idx=None, hidden=None, targets=None):
        x = self.tok_emb(idx) if self.tok_emb is not None else hidden
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        if self.lm_head is not None:
            logits, _ = self.lm_head(self.norm(x))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
            return logits, loss
        return x


# ── PP forward/backward ────────────────────────────────────────────────────

def pp_forward_backward(model, idx, targets, cfg):
    """Naive PP: sequential fwd then bwd with p2p send/recv."""
    device = next(model.parameters()).device
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_ranks = dist.get_process_group_ranks(pp_group)
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    # Forward
    if mpu.is_pipeline_first_stage():
        out = model(idx=idx)
    else:
        buf = torch.empty(idx.shape[0], idx.shape[1], cfg.dim, device=device)
        dist.recv(buf, src=pp_ranks[pp_rank - 1])
        buf.requires_grad_()
        out = model(hidden=buf)

    if not mpu.is_pipeline_last_stage():
        dist.send(out.contiguous(), dst=pp_ranks[pp_rank + 1])

    # Backward
    loss = None
    if mpu.is_pipeline_last_stage():
        logits, loss = out
        loss.backward()
    else:
        grad = torch.empty_like(out)
        dist.recv(grad, src=pp_ranks[pp_rank + 1])
        out.backward(grad)

    if not mpu.is_pipeline_first_stage():
        dist.send(buf.grad.contiguous(), dst=pp_ranks[pp_rank - 1])

    return loss


# ── Build model with FSDP ──────────────────────────────────────────────────

def build_model(cfg: QwenConfig, tp=1, cp=1, pp=1, device="cuda"):
    """
    Usage:
        dist.init_process_group("nccl")
        model = build_model(QwenConfig(dim=4096, n_layers=32), tp=2, cp=2, pp=4)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for batch in loader:
            loss = pp_forward_backward(model, batch["ids"], batch["labels"], cfg)
            opt.step(); opt.zero_grad()
    """
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )
    model = Qwen(cfg).to(device)

    # FSDP across DP ranks
    dp_group = mpu.get_data_parallel_group()
    if mpu.get_data_parallel_world_size() > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        for i in range(len(model.blocks)):
            model.blocks[i] = FSDP(model.blocks[i], process_group=dp_group, sharding_strategy=ShardingStrategy.FULL_SHARD)
        model = FSDP(model, process_group=dp_group, sharding_strategy=ShardingStrategy.FULL_SHARD)

    return model
