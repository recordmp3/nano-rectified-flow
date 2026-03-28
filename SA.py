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
    """Q,K,V: (L, d) → O: (L, d), cache: (P, Q, K, V, scale)"""
    scale = math.sqrt(K.shape[-1])
    S = Q @ K.T / scale          # (L, L)
    P = softmax_fwd_np(S)        # (L, L)
    O = P @ V                    # (L, d)
    return O, (P, Q, K, V, scale)


def attention_bwd_np(dO, cache):
    """dO: (L, d) → dQ, dK, dV"""
    P, Q, K, V, scale = cache
    dV = P.T @ dO                # (L, d)
    dP = dO @ V.T                # (L, L)
    dS = softmax_bwd_np(P, dP)   # (L, L)
    dQ = dS @ K / scale          # (L, d)
    dK = dS.T @ Q / scale        # (L, d)
    return dQ, dK, dV


# ── PyTorch version ─────────────────────────────────────────────────────────

def attention_fwd_pt(Q, K, V):
    """Q,K,V: (L, d) → O: (L, d), cache: (P, Q, K, V, scale)"""
    scale = math.sqrt(K.shape[-1])
    S = Q @ K.T / scale
    P = softmax_fwd_pt(S)
    O = P @ V
    return O, (P, Q, K, V, scale)


def attention_bwd_pt(dO, cache):
    """dO: (L, d) → dQ, dK, dV"""
    P, Q, K, V, scale = cache
    dV = P.T @ dO
    dP = dO @ V.T
    dS = softmax_bwd_pt(P, dP)
    dQ = dS @ K / scale
    dK = dS.T @ Q / scale
    return dQ, dK, dV


# ── Verify: numpy vs pytorch vs autograd ────────────────────────────────────

if __name__ == "__main__":
    L, d = 4, 8
    rng = np.random.default_rng(42)
    qn, kn, vn = [rng.standard_normal((L, d)).astype(np.float64) for _ in range(3)]

    # NumPy
    O_np, cache_np = attention_fwd_np(qn, kn, vn)
    dO_np = rng.standard_normal((L, d)).astype(np.float64)
    dQ_np, dK_np, dV_np = attention_bwd_np(dO_np, cache_np)

    # PyTorch manual
    qt, kt, vt = [torch.tensor(x) for x in (qn, kn, vn)]
    dO_t = torch.tensor(dO_np)
    O_pt, cache_pt = attention_fwd_pt(qt, kt, vt)
    dQ_pt, dK_pt, dV_pt = attention_bwd_pt(dO_t, cache_pt)

    # PyTorch autograd (ground truth)
    qa, ka, va = [torch.tensor(x, requires_grad=True) for x in (qn, kn, vn)]
    scale = math.sqrt(d)
    O_auto = torch.softmax(qa @ ka.T / scale, dim=-1) @ va
    O_auto.backward(dO_t)

    print("fwd match (np vs pt):", np.allclose(O_np, O_pt.numpy(), atol=1e-12))
    print("fwd match (pt vs autograd):", torch.allclose(O_pt, O_auto, atol=1e-12))
    print("dQ match:", np.allclose(dQ_np, qa.grad.numpy(), atol=1e-12))
    print("dK match:", np.allclose(dK_np, ka.grad.numpy(), atol=1e-12))
    print("dV match:", np.allclose(dV_np, va.grad.numpy(), atol=1e-12))
