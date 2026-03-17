"""
Minimal Qwen-like LLM with Megatron-style TP + CP + PP, combined with FSDP.

Parallelism layout (inner → outer): [TP, CP, PP, DP]
  - TP  (Tensor Parallel):   split heads / FFN across TP ranks
  - CP  (Context Parallel):  split sequence across CP ranks, all-gather KV
  - PP  (Pipeline Parallel): split layers across PP stages, p2p send/recv
  - DP  (Data Parallel):     FSDP shards params/grads/optim across DP ranks

Single-GPU compatible: if PS.init() is never called, everything defaults to 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass


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


@dataclass
class ParallelConfig:
    tp: int = 1
    cp: int = 1
    pp: int = 1


# ── Parallel State (singleton) ──────────────────────────────────────────────

class PS:
    """Global parallel state. Call PS.init() once after dist.init_process_group()."""
    tp = cp = pp = dp = 1
    tp_rank = cp_rank = pp_rank = dp_rank = 0
    tp_group = cp_group = pp_group = dp_group = None

    @staticmethod
    def init(pcfg: ParallelConfig):
        world = dist.get_world_size()
        rank = dist.get_rank()
        tp, cp, pp = pcfg.tp, pcfg.cp, pcfg.pp
        dp = world // (tp * cp * pp)
        assert world == dp * pp * cp * tp, f"world={world} != dp*pp*cp*tp={dp}*{pp}*{cp}*{tp}"
        PS.tp, PS.cp, PS.pp, PS.dp = tp, cp, pp, dp
        PS.tp_rank = rank % tp
        PS.cp_rank = (rank // tp) % cp
        PS.pp_rank = (rank // (tp * cp)) % pp
        PS.dp_rank = rank // (tp * cp * pp)

        # Create one process group per parallelism dimension
        for name in ("tp", "cp", "pp", "dp"):
            buckets = {}
            for r in range(world):
                t = r % tp; c = (r // tp) % cp
                p = (r // (tp * cp)) % pp; d = r // (tp * cp * pp)
                key = {"tp": (d, p, c), "cp": (d, p, t),
                       "pp": (d, c, t), "dp": (p, c, t)}[name]
                buckets.setdefault(key, []).append(r)
            for ranks in buckets.values():
                g = dist.new_group(ranks)
                if rank in ranks:
                    setattr(PS, f"{name}_group", g)

    @staticmethod
    def pp_prev_rank():
        return dist.get_rank() - PS.tp * PS.cp

    @staticmethod
    def pp_next_rank():
        return dist.get_rank() + PS.tp * PS.cp


# ── TP autograd primitives ──────────────────────────────────────────────────

class _Copy(torch.autograd.Function):
    """Fwd: identity | Bwd: all-reduce."""
    @staticmethod
    def forward(ctx, x, g):
        ctx.g = g; return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=ctx.g); return grad, None


class _Reduce(torch.autograd.Function):
    """Fwd: all-reduce | Bwd: identity."""
    @staticmethod
    def forward(ctx, x, g):
        dist.all_reduce(x, group=g); return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class ColLinear(nn.Module):
    """Column-parallel: shard output dim across TP."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f // PS.tp, bias=bias)

    def forward(self, x):
        if PS.tp > 1:
            x = _Copy.apply(x, PS.tp_group)
        return self.linear(x)


class RowLinear(nn.Module):
    """Row-parallel: shard input dim across TP, all-reduce output."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_f // PS.tp, out_f, bias=bias)

    def forward(self, x):
        y = self.linear(x)
        if PS.tp > 1:
            y = _Reduce.apply(y, PS.tp_group)
        return y


# ── CP autograd primitive ───────────────────────────────────────────────────

class _AllGatherCP(torch.autograd.Function):
    """Fwd: all-gather along seq dim (dim=2) | Bwd: chunk back to local."""
    @staticmethod
    def forward(ctx, x):
        ctx.cp, ctx.rank = PS.cp, PS.cp_rank
        parts = [torch.empty_like(x) for _ in range(PS.cp)]
        dist.all_gather(parts, x.contiguous(), group=PS.cp_group)
        return torch.cat(parts, dim=2)

    @staticmethod
    def backward(ctx, grad):
        return grad.chunk(ctx.cp, dim=2)[ctx.rank].contiguous()


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
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# ── Attention (TP + CP + RoPE + GQA) ────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        assert cfg.n_kv_head % PS.tp == 0, "n_kv_head must be divisible by TP"
        self.n_head = cfg.n_head // PS.tp
        self.n_kv = cfg.n_kv_head // PS.tp
        self.hd = cfg.dim // cfg.n_head
        self.wq = ColLinear(cfg.dim, cfg.dim, bias=True)
        self.wk = ColLinear(cfg.dim, self.hd * cfg.n_kv_head, bias=True)
        self.wv = ColLinear(cfg.dim, self.hd * cfg.n_kv_head)
        self.wo = RowLinear(cfg.dim, cfg.dim)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv, self.hd).transpose(1, 2)

        # RoPE with CP offset (each CP rank holds a different chunk of the seq)
        off = PS.cp_rank * L
        q = apply_rope(q, cos[off:off + L], sin[off:off + L])
        k = apply_rope(k, cos[off:off + L], sin[off:off + L])

        # CP: all-gather K, V so every rank sees the full context
        if PS.cp > 1:
            k = _AllGatherCP.apply(k)
            v = _AllGatherCP.apply(v)

        # GQA: expand KV heads
        rep = self.n_head // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Flash attention
        if PS.cp <= 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Q covers global positions [off, off+L), K covers [0, L*cp)
            full_L = L * PS.cp
            qi = torch.arange(off, off + L, device=x.device)
            ki = torch.arange(full_L, device=x.device)
            mask = qi[:, None] >= ki[None, :]  # (L, full_L) causal
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


# ── SwiGLU FFN (TP) ─────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.gate = ColLinear(cfg.dim, cfg.ffn_hidden)
        self.up = ColLinear(cfg.dim, cfg.ffn_hidden)
        self.down = RowLinear(cfg.ffn_hidden, cfg.dim)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


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

        # PP: each stage owns a contiguous slice of layers
        per = (cfg.n_layers + PS.pp - 1) // PS.pp
        self.l0 = PS.pp_rank * per
        self.l1 = min(self.l0 + per, cfg.n_layers)

        # First PP stage owns embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim) if PS.pp_rank == 0 else None

        # Local transformer blocks only
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(self.l1 - self.l0)])

        # Last PP stage owns norm + lm_head
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps) if PS.pp_rank == PS.pp - 1 else None
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False) if PS.pp_rank == PS.pp - 1 else None

        # Weight tying (only possible when embedding & lm_head on same rank, i.e. PP=1)
        if self.tok_emb is not None and self.lm_head is not None:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE buffers
        cos, sin = precompute_rope(cfg.dim // cfg.n_head, cfg.max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, idx=None, hidden=None, targets=None):
        """
        First PP stage:  pass idx   → returns hidden
        Middle PP stage: pass hidden → returns hidden
        Last PP stage:   pass hidden (+ targets for loss) → returns (logits, loss)
        PP=1:            pass idx    (+ targets for loss) → returns (logits, loss)
        """
        if self.tok_emb is not None:
            x = self.tok_emb(idx)
        else:
            x = hidden

        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)

        if self.lm_head is not None:
            logits = self.lm_head(self.norm(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return x  # intermediate/first stage: pass hidden to next stage


# ── PP forward/backward (naive, no microbatch interleaving) ─────────────────

def pp_forward_backward(model, idx, targets, cfg):
    """
    Execute one forward + backward across all PP stages via p2p send/recv.
    Each rank calls this; communication happens automatically.

    Returns loss (on last stage) or None (on other stages).
    """
    device = next(model.parameters()).device

    # ── Forward ──
    if PS.pp_rank == 0:
        out = model(idx=idx)
    else:
        buf = torch.empty(idx.shape[0], idx.shape[1], cfg.dim,
                          device=device, dtype=torch.float32)
        dist.recv(buf, src=PS.pp_prev_rank())
        buf.requires_grad_()
        out = model(hidden=buf)

    if PS.pp_rank < PS.pp - 1:
        # Send hidden to next stage
        dist.send(out.contiguous(), dst=PS.pp_next_rank())

    # ── Backward ──
    loss = None
    if PS.pp_rank == PS.pp - 1:
        logits, loss = out
        loss.backward()
    else:
        # Receive gradient from next stage
        grad = torch.empty_like(out)
        dist.recv(grad, src=PS.pp_next_rank())
        out.backward(grad)

    if PS.pp_rank > 0:
        # Send input gradient to previous stage
        dist.send(buf.grad.contiguous(), dst=PS.pp_prev_rank())

    return loss


# ── Build model with FSDP ───────────────────────────────────────────────────

def build_model(cfg: QwenConfig, pcfg: ParallelConfig = None, device="cuda"):
    """
    Initialize parallel state, create model, wrap with FSDP.

    Usage:
        dist.init_process_group(backend="nccl")
        model = build_model(QwenConfig(), ParallelConfig(tp=2, cp=2, pp=2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for batch in dataloader:
            # If PP > 1, use pp_forward_backward(); else use model() directly.
            if PS.pp > 1:
                loss = pp_forward_backward(model, batch["input_ids"], batch["labels"], cfg)
            else:
                _, loss = model(batch["input_ids"], targets=batch["labels"])
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    """
    if pcfg is not None and dist.is_initialized():
        PS.init(pcfg)

    model = Qwen(cfg).to(device)

    # FSDP: shard across DP ranks (same PP stage, same TP shard, same CP chunk)
    if PS.dp > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        # Wrap each block individually for better memory efficiency
        for i in range(len(model.blocks)):
            model.blocks[i] = FSDP(
                model.blocks[i],
                process_group=PS.dp_group,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
            )
        model = FSDP(
            model,
            process_group=PS.dp_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )

    return model
