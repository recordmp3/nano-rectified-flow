"""
Minimal Qwen-like LLM — pure Megatron-Core.

Everything (RoPE, RMSNorm, GQA, SwiGLU, TP, CP, PP) is handled by
megatron.core.models.gpt.GPTModel.  FSDP wraps the DP dimension.

    pip install megatron-core
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from functools import partial

from megatron.core import parallel_state as mpu
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func


# ── Config ──────────────────────────────────────────────────────────────────

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


def _to_megatron(cfg: QwenConfig) -> TransformerConfig:
    """Convert QwenConfig → Megatron TransformerConfig."""
    return TransformerConfig(
        num_layers=cfg.n_layers,
        hidden_size=cfg.dim,
        num_attention_heads=cfg.n_head,
        num_query_groups=cfg.n_kv_head,        # GQA
        ffn_hidden_size=cfg.ffn_hidden,
        max_position_embeddings=cfg.max_len,
        normalization="RMSNorm",
        layernorm_epsilon=cfg.norm_eps,
        position_embedding_type="rope",
        rotary_base=10000,
        activation_func=F.silu,
        gated_linear_unit=True,                 # SwiGLU
        bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
    )


# ── Build model ─────────────────────────────────────────────────────────────

def build_model(cfg: QwenConfig, tp=1, cp=1, pp=1, device="cuda"):
    """
    Usage:
        dist.init_process_group("nccl")
        cfg = QwenConfig(dim=4096, n_layers=32)
        model = build_model(cfg, tp=2, cp=2, pp=4)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for batch in loader:
            losses = train_step(model, iter([batch]), cfg)
            opt.step(); opt.zero_grad()
    """
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )

    mcfg = _to_megatron(cfg)
    model = GPTModel(
        config=mcfg,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=cfg.vocab_size,
        max_sequence_length=cfg.max_len,
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    ).to(device)

    # FSDP across DP ranks
    if mpu.get_data_parallel_world_size() > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        dp = mpu.get_data_parallel_group()
        model = FSDP(model, process_group=dp, sharding_strategy=ShardingStrategy.FULL_SHARD)

    return model


# ── PP-aware train step (Megatron schedule) ─────────────────────────────────

def _loss_func(labels, output):
    logits = output.contiguous().float()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss, {"lm_loss": loss}


def _forward_step(data_iterator, model):
    batch = next(data_iterator)
    tokens, labels = batch["input_ids"], batch["labels"]
    position_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand_as(tokens)
    output = model(tokens, position_ids, attention_mask=None)
    return output, partial(_loss_func, labels)


def train_step(model, data_iterator, cfg: QwenConfig, num_microbatches=1):
    """One fwd+bwd step.  Megatron handles PP schedule + p2p comm internally."""
    fwd_bwd = get_forward_backward_func()
    losses = fwd_bwd(
        forward_step_func=_forward_step,
        data_iterator=data_iterator,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=cfg.max_len,
        micro_batch_size=1,
        forward_only=False,
    )
    return losses
