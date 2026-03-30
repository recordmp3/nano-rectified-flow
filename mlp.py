"""Minimal MLP forward/backward with manual data parallelism via all-reduce."""

import torch
import torch.distributed as dist
import math


# ── Activations ─────────────────────────────────────────────────────────────

def relu_fwd(x):
    return x.clamp(min=0), (x > 0)              # out, mask

def relu_bwd(dout, mask):
    return dout * mask


# ── Linear forward/backward ────────────────────────────────────────────────

def linear_fwd(x, W, b):
    """x: (B, in) W: (out, in) b: (out,) → y: (B, out)"""
    return x @ W.T + b, (x, W)                  # out, cache

def linear_bwd(dout, cache):
    """dout: (B, out) → dx, dW, db"""
    x, W = cache
    dx = dout @ W                                # (B, in)
    dW = dout.T @ x                              # (out, in)
    db = dout.sum(dim=0)                         # (out,)
    return dx, dW, db


# ── 2-layer MLP: forward & backward ────────────────────────────────────────

def mlp_fwd(x, W1, b1, W2, b2):
    """x: (B, d_in) → y: (B, d_out)"""
    h_pre, cache1 = linear_fwd(x, W1, b1)       # (B, hidden)
    h, mask = relu_fwd(h_pre)                    # (B, hidden)
    y, cache2 = linear_fwd(h, W2, b2)            # (B, d_out)
    return y, (cache1, mask, cache2)


def mlp_bwd(dy, cache):
    """dy: (B, d_out) → grads for W1, b1, W2, b2"""
    cache1, mask, cache2 = cache
    dh, dW2, db2 = linear_bwd(dy, cache2)        # backward through layer 2
    dh_pre = relu_bwd(dh, mask)                  # backward through relu
    dx, dW1, db1 = linear_bwd(dh_pre, cache1)    # backward through layer 1
    return dx, dW1, db1, dW2, db2


# ── Manual data-parallel training step ──────────────────────────────────────

def dp_train_step(x, target, params, lr=1e-2):
    """
    One training step with manual data parallelism.
    Each rank holds full params + a local micro-batch.
    Gradients are all-reduced across ranks before update.
    """
    W1, b1, W2, b2 = params

    # Forward
    y, cache = mlp_fwd(x, W1, b1, W2, b2)

    # MSE loss: L = mean((y - target)^2)
    diff = y - target
    loss = (diff ** 2).mean()
    dy = 2.0 * diff / diff.numel()               # dL/dy

    # Backward
    _, dW1, db1, dW2, db2 = mlp_bwd(dy, cache)

    # All-reduce gradients across DP ranks (average)
    grads = [dW1, db1, dW2, db2]
    if dist.is_initialized():
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.AVG)

    # SGD update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return loss.item()


# ── Verify: manual backward vs autograd ─────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, d_in, d_hid, d_out = 4, 8, 16, 3

    x = torch.randn(B, d_in, dtype=torch.float64)
    W1 = torch.randn(d_hid, d_in, dtype=torch.float64)
    b1 = torch.randn(d_hid, dtype=torch.float64)
    W2 = torch.randn(d_out, d_hid, dtype=torch.float64)
    b2 = torch.randn(d_out, dtype=torch.float64)

    # Manual forward + backward
    y, cache = mlp_fwd(x, W1, b1, W2, b2)
    target = torch.randn_like(y)
    diff = y - target
    loss = (diff ** 2).mean()
    dy = 2.0 * diff / diff.numel()
    _, dW1, db1, dW2, db2 = mlp_bwd(dy, cache)

    # Autograd (ground truth)
    W1a = W1.clone().requires_grad_(True)
    b1a = b1.clone().requires_grad_(True)
    W2a = W2.clone().requires_grad_(True)
    b2a = b2.clone().requires_grad_(True)
    ha = (x @ W1a.T + b1a).clamp(min=0)
    ya = ha @ W2a.T + b2a
    loss_a = ((ya - target) ** 2).mean()
    loss_a.backward()

    print("fwd match:", torch.allclose(y, ya, atol=1e-12))
    print("dW1 match:", torch.allclose(dW1, W1a.grad, atol=1e-12))
    print("db1 match:", torch.allclose(db1, b1a.grad, atol=1e-12))
    print("dW2 match:", torch.allclose(dW2, W2a.grad, atol=1e-12))
    print("db2 match:", torch.allclose(db2, b2a.grad, atol=1e-12))
