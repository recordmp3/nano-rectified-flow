"""Minimal FSDP wrapper for nano_qwen."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from nano_qwen import Qwen, QwenConfig


def build_model(cfg: QwenConfig = QwenConfig(), device="cuda"):
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    model = Qwen(cfg).to(device)
    for i in range(len(model.blocks)):
        model.blocks[i] = FSDP(model.blocks[i], sharding_strategy=ShardingStrategy.FULL_SHARD)
    return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
