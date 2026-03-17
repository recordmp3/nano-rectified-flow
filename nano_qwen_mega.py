"""
Megatron-Core wrapper for nano_qwen: adds TP + CP + PP.

Replaces nn.Linear → ColumnParallelLinear / RowParallelLinear,
nn.Embedding → VocabParallelEmbedding, and adds CP all-gather for KV
and PP layer splitting with p2p send/recv.

    torchrun --nproc_per_node=8 train.py  # tp=2, cp=2, pp=2 → dp=1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from nano_qwen import QwenConfig, RMSNorm, precompute_rope, apply_rope


# ── CP: all-gather KV along seq dim ────────────────────────────────────────

class _CPGather(torch.autograd.Function):
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


# ── Attention (TP + CP) ────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        tp = mpu.get_tensor_model_parallel_world_size()
        self.n_head = cfg.n_head // tp
        self.n_kv = cfg.n_kv_head // tp
        self.hd = cfg.dim // cfg.n_head
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

        # RoPE with CP position offset
        cp = mpu.get_context_parallel_world_size()
        off = mpu.get_context_parallel_rank() * L
        q = apply_rope(q, cos[off:off + L], sin[off:off + L])
        k = apply_rope(k, cos[off:off + L], sin[off:off + L])

        # CP: all-gather full KV
        if cp > 1:
            k, v = _CPGather.apply(k), _CPGather.apply(v)

        # GQA expand
        rep = self.n_head // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Attention (causal mask accounts for CP position offset)
        if cp <= 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            qi = torch.arange(off, off + L, device=x.device)
            ki = torch.arange(L * cp, device=x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=qi[:, None] >= ki[None, :])

        out, _ = self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))
        return out


# ── SwiGLU FFN (TP) ────────────────────────────────────────────────────────

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


# ── Block ───────────────────────────────────────────────────────────────────

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


# ── Qwen with PP ───────────────────────────────────────────────────────────

class QwenMega(nn.Module):
    def __init__(self, cfg: QwenConfig = QwenConfig()):
        super().__init__()
        self.cfg = cfg
        pp = mpu.get_pipeline_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        per = (cfg.n_layers + pp - 1) // pp
        self.l0, self.l1 = pp_rank * per, min(pp_rank * per + per, cfg.n_layers)

        self.tok_emb = VocabParallelEmbedding(cfg.vocab_size, cfg.dim) \
            if mpu.is_pipeline_first_stage() else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(self.l1 - self.l0)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps) \
            if mpu.is_pipeline_last_stage() else None
        self.lm_head = ColumnParallelLinear(cfg.dim, cfg.vocab_size, bias=False) \
            if mpu.is_pipeline_last_stage() else None

        cos, sin = precompute_rope(cfg.dim // cfg.n_head, cfg.max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, idx=None, hidden=None, targets=None):
        x = self.tok_emb(idx) if self.tok_emb is not None else hidden
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        if self.lm_head is not None:
            logits, _ = self.lm_head(self.norm(x))
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            ) if targets is not None else None
            return logits, loss
        return x


# ── PP forward/backward ────────────────────────────────────────────────────

def pp_forward_backward(model, idx, targets, cfg):
    device = next(model.parameters()).device
    pp_ranks = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())
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


# ── Entry point ─────────────────────────────────────────────────────────────

def build_model(cfg: QwenConfig, tp=1, cp=1, pp=1, device="cuda"):
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )
    return QwenMega(cfg).to(device)
