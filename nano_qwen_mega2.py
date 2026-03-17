"""
nano_qwen, pure Megatron-Core edition.

Model definition = GPTModel, parallelism = built-in TP + CP + PP.
No hand-written Attention/FFN/RMSNorm/RoPE — all from megatron.core.

    pip install megatron-core
    torchrun --nproc_per_node=8 train.py
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


def _to_mcore(c: QwenConfig) -> TransformerConfig:
    return TransformerConfig(
        num_layers=c.n_layers,
        hidden_size=c.dim,
        num_attention_heads=c.n_head,
        num_query_groups=c.n_kv_head,
        ffn_hidden_size=c.ffn_hidden,
        max_position_embeddings=c.max_len,
        normalization="RMSNorm",
        layernorm_epsilon=c.norm_eps,
        position_embedding_type="rope",
        rotary_base=10000,
        activation_func=F.silu,
        gated_linear_unit=True,
        bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
    )


# ── build ───────────────────────────────────────────────────────────────────

def build_model(cfg: QwenConfig, tp=1, cp=1, pp=1, device="cuda"):
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )
    return GPTModel(
        config=_to_mcore(cfg),
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=cfg.vocab_size,
        max_sequence_length=cfg.max_len,
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    ).to(device)


# ── train step (PP schedule handled by megatron) ───────────────────────────

def _loss_func(labels, output):
    logits = output.contiguous().float()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss, {"lm_loss": loss}


def _forward_step(data_iter, model):
    batch = next(data_iter)
    tokens, labels = batch["input_ids"], batch["labels"]
    pos = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand_as(tokens)
    output = model(tokens, pos, attention_mask=None)
    return output, partial(_loss_func, labels)


def train_step(model, data_iter, cfg: QwenConfig, num_microbatches=1):
    """One fwd+bwd. Megatron handles PP p2p comm + schedule internally."""
    return get_forward_backward_func()(
        forward_step_func=_forward_step,
        data_iterator=data_iter,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=cfg.max_len,
        micro_batch_size=1,
        forward_only=False,
    )
