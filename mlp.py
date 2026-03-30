"""
Minimal MLP forward/backward with manual FSDP.

Each rank holds a 1/N shard of every parameter.
Before compute: all-gather to reconstruct full param.
After backward:  reduce-scatter to get sharded grad, then local SGD update.

    torchrun --nproc_per_node=4 mlp.py
"""

import torch
import torch.distributed as dist


# ── Activations ─────────────────────────────────────────────────────────────

def relu_fwd(x):
    return x.clamp(min=0), (x > 0)

def relu_bwd(dout, mask):
    return dout * mask


# ── FSDP primitives ─────────────────────────────────────────────────────────

def all_gather_param(shard):
    """Gather shards from all ranks → full param."""
    world = dist.get_world_size()
    parts = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(parts, shard.contiguous())
    return torch.cat(parts, dim=0)


def reduce_scatter_grad(full_grad):
    """Reduce full grad across ranks, each rank gets its shard."""
    world = dist.get_world_size()
    chunks = list(full_grad.chunk(world, dim=0))
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.AVG)
    return out


def shard_param(param):
    """Split param along dim=0 and return local shard."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    return param.chunk(world, dim=0)[rank].clone()


# ── FSDP linear: gather → compute → discard full param ─────────────────────

def linear_fwd(x, W_shard, b):
    """x: (B, in), W_shard: (out/N, in), b: (out,) → y: (B, out)"""
    W_full = all_gather_param(W_shard)             # reconstruct full W
    y = x @ W_full.T + b
    return y, (x, W_full)                          # cache full W for bwd


def linear_bwd(dout, cache):
    """→ dx, dW_shard, db"""
    x, W_full = cache
    dx = dout @ W_full                             # (B, in)
    dW_full = dout.T @ x                           # (out, in)
    dW_shard = reduce_scatter_grad(dW_full)        # each rank gets its shard
    db = dout.sum(dim=0)
    # db is duplicated on all ranks, average it
    dist.all_reduce(db, op=dist.ReduceOp.AVG)
    return dx, dW_shard, db


# ── 2-layer MLP ─────────────────────────────────────────────────────────────

def mlp_fwd(x, W1s, b1, W2s, b2):
    h_pre, cache1 = linear_fwd(x, W1s, b1)
    h, mask = relu_fwd(h_pre)
    y, cache2 = linear_fwd(h, W2s, b2)
    return y, (cache1, mask, cache2)


def mlp_bwd(dy, cache):
    cache1, mask, cache2 = cache
    dh, dW2s, db2 = linear_bwd(dy, cache2)
    dh_pre = relu_bwd(dh, mask)
    dx, dW1s, db1 = linear_bwd(dh_pre, cache1)
    return dx, dW1s, db1, dW2s, db2


# ── FSDP train step ────────────────────────────────────────────────────────

def fsdp_train_step(x, target, shards, lr=1e-2):
    """
    shards = [W1_shard, b1, W2_shard, b2]
    W shards are (out/N, in), biases are full (not sharded, small).
    """
    W1s, b1, W2s, b2 = shards

    # Forward (all-gather inside)
    y, cache = mlp_fwd(x, W1s, b1, W2s, b2)

    # MSE loss
    diff = y - target
    loss = (diff ** 2).mean()
    dy = 2.0 * diff / diff.numel()

    # Backward (reduce-scatter inside)
    _, dW1s, db1, dW2s, db2 = mlp_bwd(dy, cache)

    # SGD on shards directly
    W1s -= lr * dW1s
    b1  -= lr * db1
    W2s -= lr * dW2s
    b2  -= lr * db2

    return loss.item()


# ── Verify ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)

    B, d_in, d_hid, d_out = 4, 8, 16, 4

    # Create full params on all ranks (same seed → same init)
    W1 = torch.randn(d_hid, d_in, dtype=torch.float64)
    b1 = torch.randn(d_hid, dtype=torch.float64)
    W2 = torch.randn(d_out, d_hid, dtype=torch.float64)
    b2 = torch.randn(d_out, dtype=torch.float64)
    x = torch.randn(B, d_in, dtype=torch.float64)
    target = torch.randn(B, d_out, dtype=torch.float64)

    # Each rank shards W along dim=0
    W1s = shard_param(W1)   # (d_hid/N, d_in)
    W2s = shard_param(W2)   # (d_out/N, d_hid)
    b1c, b2c = b1.clone(), b2.clone()

    # FSDP step
    loss = fsdp_train_step(x, target, [W1s, b1c, W2s, b2c], lr=0.01)

    # Ground truth: full-param autograd on rank 0
    W1a = W1.clone().requires_grad_(True)
    b1a = b1.clone().requires_grad_(True)
    W2a = W2.clone().requires_grad_(True)
    b2a = b2.clone().requires_grad_(True)
    ha = (x @ W1a.T + b1a).clamp(min=0)
    ya = ha @ W2a.T + b2a
    loss_a = ((ya - target) ** 2).mean()
    loss_a.backward()

    # After SGD
    W1_ref = W1 - 0.01 * W1a.grad
    W2_ref = W2 - 0.01 * W2a.grad

    # Check: my shard should match the corresponding chunk of the reference
    W1_ref_shard = W1_ref.chunk(world, dim=0)[rank]
    W2_ref_shard = W2_ref.chunk(world, dim=0)[rank]

    if rank == 0:
        print(f"world_size={world}")
    print(f"[rank {rank}] W1 shard match: {torch.allclose(W1s, W1_ref_shard, atol=1e-12)}")
    print(f"[rank {rank}] W2 shard match: {torch.allclose(W2s, W2_ref_shard, atol=1e-12)}")

    dist.destroy_process_group()
