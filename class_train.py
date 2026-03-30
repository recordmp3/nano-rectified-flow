"""
Minimal CNN classification training on CIFAR-10.
面试 debug 专用：注释标注了所有常见 bug 插入点。

Usage: python class_train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ⚠️ BUG点: Conv2d 参数顺序是 (in_channels, out_channels, kernel_size)
        #    常见错误: 写反 in/out channels, 例如 Conv2d(32, 3, 3) 把输入输出写反
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # 3通道输入(RGB), 32输出
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 32输入, 64输出

        # ⚠️ BUG点: BatchNorm 的参数是 num_features, 必须和对应 conv 的 out_channels 一致
        #    常见错误: bn1 写成 BatchNorm2d(3) 或 BatchNorm2d(64), 通道数不匹配
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        # ⚠️ BUG点: Linear 的 in_features 必须精确匹配 flatten 后的维度
        #    CIFAR图片 32x32 → conv1+pool → 16x16 → conv2+pool → 8x8
        #    所以 flatten 后是 64 * 8 * 8 = 4096
        #    常见错误: 写成 64*16*16(忘了pool), 64*7*7(用了ImageNet尺寸),
        #             或 32*8*8(用了conv1的通道数)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # ⚠️ BUG点: Dropout 放在 __init__ 里定义 vs forward 里调用 F.dropout
        #    如果用 F.dropout 忘了传 training=self.training, eval时也会 drop
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # ⚠️ BUG点: conv → bn → relu → pool 的顺序
        #    常见错误1: 把 bn 放在 relu 之后 (可以work但不标准, 面试官可能问为什么)
        #    常见错误2: 漏掉 relu, 变成纯线性网络
        #    常见错误3: pool 放在 conv 之前 (尺寸直接崩掉)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B,3,32,32) → (B,32,16,16)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B,32,16,16) → (B,64,8,8)

        # ⚠️ BUG点: flatten 的方式
        #    常见错误1: x.view(x.size(0), -1) 写成 x.view(-1) 把 batch 维也拍平了
        #    常见错误2: x.reshape(B, 64*8*8) 但 B 没定义或硬编码了一个固定值
        x = x.view(x.size(0), -1)                        # (B, 64*8*8) = (B, 4096)

        # ⚠️ BUG点: dropout 的位置
        #    常见错误: dropout 放在最后一层输出之后(loss之前), 会让训练不稳定
        x = self.dropout(F.relu(self.fc1(x)))             # (B, 128)

        # ⚠️ BUG点: 最后一层不要加 softmax/relu!
        #    CrossEntropyLoss 内部自带 log_softmax
        #    常见错误1: 加了 F.softmax → 做了两次 softmax, 梯度会非常小
        #    常见错误2: 加了 F.relu → 负 logit 全变0, 分类崩掉
        x = self.fc2(x)                                  # (B, num_classes) raw logits
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size=128):
    # ⚠️ BUG点: transforms 的顺序
    #    ToTensor() 必须在 Normalize() 之前, 因为 Normalize 需要 tensor 输入
    #    常见错误: 把 Normalize 放在 ToTensor 前面, 报错 PIL Image 没有 sub 方法
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # ⚠️ BUG点: RandomCrop 需要先 padding 再 crop 回原尺寸
        #    常见错误: RandomCrop(32) 没加 padding, 等于没做增强
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # ⚠️ BUG点: Normalize 的均值和标准差
        #    常见错误1: 均值标准差传反 → Normalize(std, mean)
        #    常见错误2: 用 ImageNet 的值 ([0.485,0.456,0.406]) 而不是 CIFAR-10 的值
        #    常见错误3: 只传一个标量 → 三个通道用同一个值
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        # ⚠️ BUG点: 测试集不要做 RandomHorizontalFlip / RandomCrop!
        #    常见错误: train 和 test 用同一个 transform, 测试时也做了随机增强
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # ⚠️ BUG点: DataLoader 参数
    #    常见错误1: 训练集 shuffle=False → 每个 epoch 顺序一样, 模型学不好
    #    常见错误2: 测试集 shuffle=True → 不影响结果但不规范
    #    常见错误3: drop_last=True 在测试集 → 丢掉最后一个 batch, 指标不准
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    # ⚠️ BUG点: model.train() vs model.eval()
    #    常见错误: 忘了 model.train() → BatchNorm 用的是 running stats 而非 batch stats
    #    Dropout 也不会生效
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        # ⚠️ BUG点: 数据和模型必须在同一个 device
        #    常见错误: 忘了 .to(device), 模型在 GPU 数据在 CPU
        inputs, targets = inputs.to(device), targets.to(device)

        # ⚠️⚠️⚠️ BUG重灾区: optimizer.zero_grad() 的位置
        #    正确: 在 forward 之前清零
        #    常见错误1: 放在 loss.backward() 之后 → 梯度刚算完就清掉了, 等于没更新
        #    常见错误2: 完全忘写 → 梯度累积, loss 爆炸
        #    常见错误3: 放在 optimizer.step() 之后 → 功能上可以, 但不如放前面清晰
        optimizer.zero_grad()

        # ⚠️ BUG点: forward 调用
        #    常见错误: 写成 model.forward(inputs) 而不是 model(inputs)
        #    直接调 forward 会跳过 hooks 和 nn.Module 的其他机制
        outputs = model(inputs)

        # ⚠️ BUG点: loss 函数的输入顺序
        #    CrossEntropyLoss 是 (predictions, targets) 不是 (targets, predictions)
        #    常见错误: criterion(targets, outputs) → targets 被当成 logits, 直接崩
        loss = criterion(outputs, targets)

        # ⚠️⚠️⚠️ BUG重灾区: backward + step 的顺序
        #    必须是: zero_grad → forward → loss → backward → step
        #    常见错误1: 先 step 再 backward → 用的是上一轮的梯度
        #    常见错误2: 忘了 backward → step 用的是零梯度, 参数不动
        #    常见错误3: backward 调了两次 → 梯度翻倍 (除非 retain_graph=True)
        loss.backward()
        optimizer.step()

        # ⚠️ BUG点: loss.item()
        #    常见错误: 用 loss 而不是 loss.item() 累加 → tensor 不释放, 显存泄漏
        total_loss += loss.item() * inputs.size(0)

        # ⚠️ BUG点: 计算准确率
        #    常见错误1: argmax 的 dim 写错, dim=0 是 batch 维度, 应该用 dim=1
        #    常见错误2: 用 softmax 再 argmax (多余, argmax 前加不加 softmax 结果一样)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()   # ⚠️ BUG点: 忘了 no_grad → eval 时也算梯度, 浪费显存
def evaluate(model, loader, criterion, device):
    # ⚠️ BUG点: 忘了 model.eval()
    #    → BatchNorm 用 batch stats 而非 running stats, 结果不稳定
    #    → Dropout 还在随机丢神经元, 每次 eval 结果不同
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ⚠️ BUG点: device 设置
    #    常见错误: 硬编码 device="cuda" 但机器没有 GPU → 直接报错
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(num_classes=10).to(device)

    # ⚠️ BUG点: CrossEntropyLoss 的注意事项
    #    1. 内部做 log_softmax + nll_loss, 所以模型最后一层不要加 softmax
    #    2. targets 必须是 long 类型 (class indices), 不是 one-hot
    #    常见错误: 用了 NLLLoss 但模型输出没有过 log_softmax → loss 是错的
    criterion = nn.CrossEntropyLoss()

    # ⚠️ BUG点: optimizer 传入的参数
    #    常见错误1: 忘了传 model.parameters() → 优化器不知道要更新什么
    #    常见错误2: lr 设太大(如 1.0) → loss 爆炸; 设太小(如 1e-8) → 不收敛
    #    常见错误3: 写成 model.parameters 而不是 model.parameters() → 传了个函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ⚠️ BUG点: scheduler 的 step 时机
    #    StepLR 应该每个 epoch 调一次 scheduler.step(), 不是每个 batch
    #    常见错误: 在 batch 循环里调 scheduler.step() → lr 衰减太快
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, test_loader = get_dataloaders(batch_size=128)

    for epoch in range(30):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # ⚠️ BUG点: scheduler.step() 放在 epoch 结束后
        #    常见错误1: 放在 train_one_epoch 之前 → 第一个 epoch 就衰减了
        #    常见错误2: 完全忘了调 → lr 永远不变
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.1f}% | "
              f"Test Loss {test_loss:.4f} Acc {test_acc:.1f}% | "
              f"LR {optimizer.param_groups[0]['lr']:.6f}")

    # ⚠️ BUG点: 保存模型
    #    常见错误1: torch.save(model, path) 保存整个模型 → 加载时必须有原始类定义
    #    推荐: torch.save(model.state_dict(), path) 只保存权重
    #    常见错误2: 忘了保存 optimizer state → 恢复训练时 lr/momentum 全丢了
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, "checkpoint.pt")


if __name__ == "__main__":
    main()
