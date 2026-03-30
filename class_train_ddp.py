"""
Minimal CNN classification training on CIFAR-10 with DDP.
面试 debug 专用：注释标注了 DDP 相关的所有常见 bug 插入点。

Usage: torchrun --nproc_per_node=4 class_train_ddp.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


# ══════════════════════════════════════════════════════════════════════════════
# Model (和单卡版完全一样)
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
# DDP setup / cleanup
# ══════════════════════════════════════════════════════════════════════════════

def setup():
    # ⚠️ BUG点: init_process_group 的 backend
    #    GPU 用 "nccl", CPU 用 "gloo"
    #    常见错误1: GPU 训练用了 "gloo" → 能跑但慢 10 倍
    #    常见错误2: CPU 环境用了 "nccl" → 直接报错 (nccl 不支持 CPU)
    dist.init_process_group("nccl")

    # ⚠️⚠️⚠️ BUG重灾区: 设置当前 GPU device
    #    torchrun 会自动设 LOCAL_RANK 环境变量
    #    常见错误1: 忘了 set_device → 所有进程挤在 GPU:0, OOM
    #    常见错误2: 用 RANK 而不是 LOCAL_RANK → 多节点时 RANK 可能 > GPU 数量
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

    # ⚠️⚠️⚠️ BUG重灾区: DDP 必须用 DistributedSampler
    #    DistributedSampler 把数据集按 rank 切分, 每个 rank 只看 1/N 的数据
    #    常见错误1: 不用 DistributedSampler → 每个 rank 跑全量数据, 等于没分布式
    #              梯度被 all-reduce 求平均后相当于 lr 被放大了 N 倍
    #    常见错误2: 训练集用了但测试集忘了 → eval 时每个 rank 算全量, 重复计算
    train_sampler = DistributedSampler(train_set, shuffle=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    # ⚠️ BUG点: 用了 DistributedSampler 后 DataLoader 的 shuffle 必须设 False
    #    因为 shuffle 由 sampler 控制, DataLoader 的 shuffle 和 sampler 互斥
    #    常见错误: shuffle=True + sampler=train_sampler → 直接报 ValueError
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
        # ⚠️ BUG点: pin_memory=True 时要用 non_blocking=True 加速 H2D 传输
        #    不加也不会错, 但浪费了 pin_memory 的优势
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        # ⚠️ BUG点: DDP 包装后, model(x) 会自动在 backward 时做 all-reduce
        #    常见错误: 在 DDP 外面又手动 all-reduce 了一次 → 梯度翻倍
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    # ⚠️ BUG点: 这里的 loss/acc 只是当前 rank 的局部值
    #    如果要打印全局准确的指标, 需要 all_reduce total_loss / correct / total
    #    不 reduce 也能跑, 但打印出来的值只代表 1/N 的数据
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

    # 聚合所有 rank 的指标 (可选但推荐)
    stats = torch.tensor([total_loss, correct, total], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[2]).item(), (100.0 * stats[1] / stats[2]).item()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    # ⚠️⚠️⚠️ BUG重灾区: 模型必须先 .to(device) 再包 DDP
    #    常见错误1: 先包 DDP 再 to(device) → DDP 不知道模型在哪个 GPU
    #    常见错误2: DDP 的 device_ids 和模型实际所在的 GPU 不一致
    model = CNN(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()

    # ⚠️ BUG点: optimizer 必须用 DDP 包装后的 model.parameters()
    #    常见错误: 在 DDP 包装前就创建了 optimizer → optimizer 持有的是原始参数
    #    DDP 包装后参数对象不变所以实际能跑, 但逻辑上应该在 DDP 之后创建
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, test_loader, train_sampler = get_dataloaders(batch_size=128)

    for epoch in range(30):
        # ⚠️⚠️⚠️ BUG重灾区: 每个 epoch 必须调 sampler.set_epoch(epoch)
        #    DistributedSampler 用 epoch 做随机种子来 shuffle
        #    常见错误: 忘了 set_epoch → 每个 epoch 数据顺序完全一样, 等于没 shuffle
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # ⚠️ BUG点: 只在 rank 0 打印, 否则 N 个进程打 N 遍
        #    常见错误: 所有 rank 都打印 → 日志混乱且重复
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
                  f"Test Loss {test_loss:.4f} Acc {test_acc:.1f}% | "
                  f"LR {optimizer.param_groups[0]['lr']:.6f}")

    # ⚠️ BUG点: 只在 rank 0 保存 checkpoint
    #    常见错误1: 所有 rank 都保存 → 文件覆盖或浪费磁盘
    #    常见错误2: 保存 model.state_dict() → key 带 "module." 前缀
    #              因为 DDP 包了一层, state_dict 的 key 变成 "module.conv1.weight" 等
    #              加载时要么用 model.module.state_dict(), 要么 load 时 strip 前缀
    if dist.get_rank() == 0:
        torch.save({
            # ⚠️ 用 model.module.state_dict() 去掉 "module." 前缀, 方便单卡加载
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, "checkpoint.pt")

    # ⚠️ BUG点: 所有 rank 必须一起到达 barrier, 否则有的 rank 提前退出导致死锁
    #    常见错误: rank 0 保存完就 cleanup, 其他 rank 还卡在某个 collective op
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
