"""
Minimal CNN classification training on CIFAR-10 with FSDP.
面试 debug 专用：注释标注了 FSDP 相关的所有常见 bug 插入点。

Usage: torchrun --nproc_per_node=4 class_train_fsdp.py
"""

import os
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torchvision import datasets, transforms


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Distributed setup
# ══════════════════════════════════════════════════════════════════════════════

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # ⚠️ BUG点: FSDP 和 DDP 一样需要 DistributedSampler
    #    常见错误: 以为 FSDP 只管参数切片不管数据 → 忘了 sampler → 每个 rank 跑全量
    train_sampler = DistributedSampler(train_set, shuffle=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                              sampler=train_sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             sampler=test_sampler, num_workers=2, pin_memory=True)
    return train_loader, test_loader, train_sampler


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # ⚠️ BUG点: FSDP 的 backward 和 DDP 不同
        #    DDP: backward 时 all-reduce 梯度
        #    FSDP: backward 时 reduce-scatter 梯度 (每个 rank 只保留自己 shard 的梯度)
        #    常见错误: backward 后手动 all-reduce → 和 FSDP 内部通信冲突
        loss.backward()

        # ⚠️ BUG点: gradient clipping 和 FSDP
        #    FSDP 下梯度是 sharded 的, 不能直接 clip_grad_norm_ 因为每个 rank 只有部分梯度
        #    必须用 model.clip_grad_norm_(max_norm) 而不是 torch.nn.utils.clip_grad_norm_
        #    常见错误: 用 nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #             → norm 算的是局部 shard 的 norm, 不是全局 norm, 结果不对
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    stats = torch.tensor([total_loss, correct, total], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[2]).item(), (100.0 * stats[1] / stats[2]).item()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    # ⚠️⚠️⚠️ BUG重灾区: FSDP 包装顺序
    #    必须: 创建模型 → .to(device) → FSDP 包装
    #    常见错误1: 先 FSDP 包装再 .to(device) → 参数已经被 shard 了, 再 move 会出错
    #    常见错误2: 忘了 .to(device), 模型在 CPU 上 → FSDP 用 NCCL 但参数在 CPU
    model = CNN(num_classes=10).to(device)

    # ⚠️ BUG点: auto_wrap_policy 控制 FSDP 的切分粒度
    #    - 不设 policy: 整个模型一个 FSDP unit, all-gather 时峰值 = 完整模型, 没省多少
    #    - size_based: 按参数量自动拆分子模块, 每个子模块单独 all-gather
    #    - lambda/自定义: 手动控制哪些模块被 wrap
    #    常见错误: min_num_params 设太大 → 没有子模块被 wrap, 退化成单 FSDP unit
    wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1000)

    # ⚠️ BUG点: MixedPrecision 配置
    #    param_dtype: 参数 all-gather 后的计算精度
    #    reduce_dtype: reduce-scatter 时梯度的精度
    #    buffer_dtype: BN running_mean/var 等 buffer 的精度
    #    常见错误1: buffer_dtype 设成 fp16 → BN 的 running stats 精度不够, 推理时输出飘
    #    常见错误2: 只设 param_dtype=fp16 不设 reduce_dtype → 梯度 reduce 也用 fp16, 精度丢失
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,    # 梯度用 fp32 聚合, 避免精度损失
        buffer_dtype=torch.float32,    # BN buffers 保持 fp32
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # ⚠️ BUG点: device_id 必须和 cuda.set_device 一致
        #    常见错误: 不传 device_id → FSDP 可能把 shard 放在错误的 GPU
        device_id=local_rank,
    )

    criterion = nn.CrossEntropyLoss()

    # ⚠️ BUG点: optimizer 必须在 FSDP 包装之后创建
    #    因为 FSDP 会 flatten + shard 参数, 包装后 model.parameters() 返回的是 FlatParameter
    #    常见错误: FSDP 包装前创建 optimizer → optimizer 持有原始参数, 和 FSDP 的不一致
    #             → 参数更新不到, loss 不降
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, test_loader, train_sampler = get_dataloaders(batch_size=128)

    for epoch in range(30):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
                  f"Test Loss {test_loss:.4f} Acc {test_acc:.1f}% | "
                  f"LR {optimizer.param_groups[0]['lr']:.6f}")

    # ⚠️⚠️⚠️ BUG重灾区: FSDP 保存 checkpoint
    #    FSDP 的参数是 sharded 的, 不能直接 model.state_dict()
    #    方法1: FULL_STATE_DICT — 所有 rank 参与聚合完整参数, rank 0 保存
    #    方法2: SHARDED_STATE_DICT — 每个 rank 保存自己的 shard (适合大模型)
    #    常见错误1: 直接 model.state_dict() 保存 → 只存了当前 rank 的 shard, 不完整
    #    常见错误2: 只在 rank 0 调 state_dict() → 死锁, 因为聚合需要所有 rank 参与
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # 所有 rank 都必须参与 (内部做 all-gather), 但只有 rank 0 拿到完整参数
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        state = model.state_dict()
        if dist.get_rank() == 0:
            torch.save({"model": state, "optimizer": optimizer.state_dict()}, "checkpoint.pt")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
