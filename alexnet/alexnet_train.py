"""
AlexNet 训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import socket
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from alexnet import alexnet

# 超参数配置

# CIFAR-10 归一化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
#IMAGENET 归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

BATCH_SIZE = 128
EPOCHS = 75  #30
LEARNING_RATE = 0.015 #0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3 #5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
LOG_DIR = os.path.join(SCRIPT_DIR, "runs", "alexnet_cifar10")
DATA_DIR = os.path.join(SCRIPT_DIR, "../_data")

os.makedirs(SAVE_DIR, exist_ok=True)

# 主训练函数

def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{LOG_DIR}_{timestamp}"
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # -------------------- 打印配置信息 --------------------
    print("=" * 95)
    print("AlexNet Training on CIFAR-10")
    print("=" * 95)
    print(f"TensorBoard URL: http://localhost:6006")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}  |  Epochs: {EPOCHS}  |  Learning Rate: {LEARNING_RATE}")
    print(f"TensorBoard Log: {log_dir}")
    print("=" * 95)
    
    # -------------------- 数据预处理 --------------------
    print("\n加载 CIFAR-10 数据集...")
    
    transform_train = transforms.Compose([
        #transforms.Resize(224),
        transforms.RandomHorizontalFlip(),

        #transforms.RandomCrop(224, padding=4),
        transforms.RandomCrop(32, padding=4),
        
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  #补充添加，用于增强数据，减缓过拟合

        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD) #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    transform_val = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD) #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # -------------------- 模型、损失函数、优化器 --------------------
    model = alexnet(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    # -------------------- 学习率调度器 --------------------
    #等步长衰减
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    # T_max：总epoch数（设为你的训练epoch总数）
    # eta_min：最小学习率，一般设为0或初始学习率的1/100

    # -------------------- TensorBoard 初始化记录 --------------------

    dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
    #dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

    writer.add_graph(model, dummy_input)

    sample_images, sample_labels = next(iter(train_loader))
    writer.add_image("Data/Sample_Image", sample_images[0], 0)

    # -------------------- 训练状态变量 --------------------
    best_acc = 0.0
    global_step = 0
    total_train_samples = len(train_dataset)
    total_val_samples = len(val_dataset)

    # -------------------- 训练循环 --------------------
    print("\n开始训练...")
    print("-" * 95)
    print(f"{'Epoch':^10} {'Train Loss':^12} {'Train Acc':^10} {'Val Loss':^10} {'Val Acc':^10} {'LR':^10} {'Time':^10} {'Samples/s':^10}")
    print("-" * 95)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            global_step += 1

        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = 100.0 * correct / total

        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / total_val_samples
        epoch_val_acc = 100.0 * correct / total

        # 计算训练效率
        epoch_time = time.time() - epoch_start_time
        samples_per_second = (total_train_samples + total_val_samples) / epoch_time

        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # -------------------- TensorBoard 记录 --------------------
        writer.add_scalar('Epoch/Train_Loss', epoch_train_loss, epoch)
        writer.add_scalar('Epoch/Train_Acc', epoch_train_acc, epoch)
        writer.add_scalar('Epoch/Val_Loss', epoch_val_loss, epoch)
        writer.add_scalar('Epoch/Val_Acc', epoch_val_acc, epoch)
        writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)

        # 每5个epoch记录权重直方图
        if epoch % 5 == 0 or epoch == 1:
            log_weight_histograms(writer, model, epoch)

        # 更新最佳准确率
        is_best = epoch_val_acc > best_acc
        if is_best:
            best_acc = epoch_val_acc

        # 打印训练进度
        print(f" [{epoch:2d}/{EPOCHS}]   {epoch_train_loss:>10.4f}   {epoch_train_acc:>8.2f}%   "
              f"{epoch_val_loss:>8.4f}   {epoch_val_acc:>8.2f}%   {current_lr:>8.6f}   "
              f"{format_time(epoch_time):>8}   {samples_per_second:>8.0f}{'  *BEST*' if is_best else ''}")

    # -------------------- 保存最终模型 --------------------
    final_checkpoint = {
        'timestamp': timestamp,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_acc,
    }
    final_path = os.path.join(SAVE_DIR, f"alexnet_{timestamp}_acc{best_acc:.1f}.pth")
    torch.save(final_checkpoint, final_path)

    best_path = os.path.join(SAVE_DIR, "best_model.pth")
    torch.save(final_checkpoint, best_path)

    # -------------------- 训练结束信息 --------------------
    print("-" * 95)
    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"本次训练保存: {final_path}")
    print(f"最佳模型链接: {best_path}")
    print(f"TensorBoard 日志: {log_dir}")
    print(f"查看可视化: tensorboard --logdir={log_dir}")
    print("=" * 95)

    writer.close()


# 工具函数

def log_weight_histograms(writer, model, epoch):
    """
    记录模型权重和梯度的直方图到 TensorBoard
    
    用于诊断:
    - 梯度消失: 梯度分布集中在 0 附近
    - 梯度爆炸: 梯度分布范围极大
    - 权重初始化问题: 权重分布不合理
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch 模型
        epoch: 当前 epoch
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"Weights/{name}", param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)


def format_time(seconds):
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串，如 "45.2s", "2m 30.5s", "1h 15m 30.0s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.1f}s"



if __name__ == '__main__':
    main()
