"""Minimal self-attention forward & backward, in NumPy and PyTorch."""

import numpy as np
import torch
import math


# ── softmax helpers ─────────────────────────────────────────────────────────

def softmax_fwd_np(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_bwd_np(p, dp):
    """dL/dx given p=softmax(x) and dp=dL/dp."""
    # Jacobian: diag(p) - p pᵀ  →  dx = p * (dp - sum(p * dp))
    s = (p * dp).sum(axis=-1, keepdims=True)
    return p * (dp - s)


def softmax_fwd_pt(x):
    e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def softmax_bwd_pt(p, dp):
    s = (p * dp).sum(dim=-1, keepdim=True)
    return p * (dp - s)


# ── NumPy version ───────────────────────────────────────────────────────────

def attention_fwd_np(Q, K, V):
    """Q,K,V: (B, L, d) → O: (B, L, d), cache: (P, Q, K, V, scale)"""
    scale = math.sqrt(K.shape[-1])
    S = Q @ K.swapaxes(-2, -1) / scale   # (B, L, L)
    P = softmax_fwd_np(S)                 # (B, L, L)
    O = P @ V                             # (B, L, d)
    return O, (P, Q, K, V, scale)


def attention_bwd_np(dO, cache):
    """dO: (B, L, d) → dQ, dK, dV"""
    P, Q, K, V, scale = cache
    dV = P.swapaxes(-2, -1) @ dO                # (B, L, d)
    dP = dO @ V.swapaxes(-2, -1)                # (B, L, L)
    dS = softmax_bwd_np(P, dP)                  # (B, L, L)
    dQ = dS @ K / scale                         # (B, L, d)
    dK = dS.swapaxes(-2, -1) @ Q / scale        # (B, L, d)
    return dQ, dK, dV


# ── PyTorch version ─────────────────────────────────────────────────────────

def attention_fwd_pt(Q, K, V):
    """Q,K,V: (B, L, d) → O: (B, L, d), cache: (P, Q, K, V, scale)"""
    scale = math.sqrt(K.shape[-1])
    S = Q @ K.transpose(-2, -1) / scale   # (B, L, L)
    P = softmax_fwd_pt(S)                  # (B, L, L)
    O = P @ V                              # (B, L, d)
    return O, (P, Q, K, V, scale)


def attention_bwd_pt(dO, cache):
    """dO: (B, L, d) → dQ, dK, dV"""
    P, Q, K, V, scale = cache
    dV = P.transpose(-2, -1) @ dO                # (B, L, d)
    dP = dO @ V.transpose(-2, -1)                # (B, L, L)
    dS = softmax_bwd_pt(P, dP)                   # (B, L, L)
    dQ = dS @ K / scale                          # (B, L, d)
    dK = dS.transpose(-2, -1) @ Q / scale        # (B, L, d)
    return dQ, dK, dV


# ── Verify: numpy vs pytorch vs autograd ────────────────────────────────────

if __name__ == "__main__":
    B, L, d = 2, 4, 8
    rng = np.random.default_rng(42)
    qn, kn, vn = [rng.standard_normal((B, L, d)).astype(np.float64) for _ in range(3)]

    # NumPy
    O_np, cache_np = attention_fwd_np(qn, kn, vn)
    dO_np = rng.standard_normal((B, L, d)).astype(np.float64)
    dQ_np, dK_np, dV_np = attention_bwd_np(dO_np, cache_np)

    # PyTorch manual
    qt, kt, vt = [torch.tensor(x) for x in (qn, kn, vn)]
    dO_t = torch.tensor(dO_np)
    O_pt, cache_pt = attention_fwd_pt(qt, kt, vt)
    dQ_pt, dK_pt, dV_pt = attention_bwd_pt(dO_t, cache_pt)

    # PyTorch autograd (ground truth)
    qa, ka, va = [torch.tensor(x, requires_grad=True) for x in (qn, kn, vn)]
    scale = math.sqrt(d)
    O_auto = torch.softmax(qa @ ka.transpose(-2, -1) / scale, dim=-1) @ va
    O_auto.backward(dO_t)

    print("fwd match (np vs pt):", np.allclose(O_np, O_pt.numpy(), atol=1e-12))
    print("fwd match (pt vs autograd):", torch.allclose(O_pt, O_auto, atol=1e-12))
    print("dQ match:", np.allclose(dQ_np, qa.grad.numpy(), atol=1e-12))
    print("dK match:", np.allclose(dK_np, ka.grad.numpy(), atol=1e-12))
    print("dV match:", np.allclose(dV_np, va.grad.numpy(), atol=1e-12))
