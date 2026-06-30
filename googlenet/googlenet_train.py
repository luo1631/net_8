"""
GoogLeNet 训练脚本
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
import yaml
import random
import numpy as np
import platform
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from googlenet import googlenet, GoogLeNetOutputs

# 超参数配置

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

BATCH_SIZE = 128
EPOCHS = 45
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
AUX_LOGITS = True
AUX_LOSS_WEIGHT = 0.3
ALPHA = 0.5    #1.0

NUM_CLASSES = 10
DROPOUT = 0.2
DROPOUT_AUX = 0.7

OPTIMIZER_TYPE = "SGD"
SCHEDULER_TYPE = "CosineAnnealingLR"
SCHEDULER_T_MAX = EPOCHS
SCHEDULER_ETA_MIN = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
LOG_DIR = os.path.join(SCRIPT_DIR, "runs", "googlenet_cifar10")
DATA_DIR = os.path.join(SCRIPT_DIR, "../_data")

# 主训练函数

def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{LOG_DIR}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # -------------------- 数据预处理 --------------------
    print("\n加载 CIFAR-10 数据集...")
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_val)

    _num_workers = 0 if platform.system() == "Windows" else 2
    _pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=_num_workers,
        pin_memory=_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin_memory
    )

    # -------------------- 模型、损失函数、优化器 --------------------
    model = googlenet(
        num_classes=NUM_CLASSES,
        aux_logits=AUX_LOGITS,
        alpha=ALPHA,
        dropout=DROPOUT,
        dropout_aux=DROPOUT_AUX
    ).to(DEVICE)

    config_groups = collect_config()
    print_config(config_groups, model, log_dir)
    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER_TYPE.upper() == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=LEARNING_RATE,
            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER_TYPE.upper() == "ADAM":
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER_TYPE.upper() == "ADAMW":
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {OPTIMIZER_TYPE}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=SCHEDULER_T_MAX, 
        eta_min=SCHEDULER_ETA_MIN
    )

    # -------------------- 生成配置文件 --------------------
    save_config_yaml(
        log_dir, model,
        num_classes=NUM_CLASSES,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        transform_train=transform_train,
        transform_val=transform_val,
        timestamp=timestamp,
    )

    # -------------------- TensorBoard 初始化记录 --------------------
    sample_images, sample_labels = next(iter(train_loader))
    writer.add_image("Data/Sample_Image", sample_images[0], 0)

    model_summary = build_model_summary_text(model)
    writer.add_text("Model/Architecture", model_summary, 0)

    hparam_dict = {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "alpha": ALPHA, "dropout": DROPOUT, "dropout_aux": DROPOUT_AUX,
        "aux_loss_weight": AUX_LOSS_WEIGHT, "num_classes": NUM_CLASSES,
    }

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
            
            if isinstance(outputs, GoogLeNetOutputs):
                loss_main = criterion(outputs.logits, labels)
                loss_aux2 = criterion(outputs.aux_logits2, labels) if outputs.aux_logits2 is not None else torch.tensor(0.0, device=DEVICE)
                loss_aux1 = criterion(outputs.aux_logits1, labels) if outputs.aux_logits1 is not None else torch.tensor(0.0, device=DEVICE)
                loss = loss_main + AUX_LOSS_WEIGHT * (loss_aux1 + loss_aux2)
                _, predicted = torch.max(outputs.logits, 1)
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
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
                # eval 模式下 forward() 直接返回主 logits 张量（不含辅助分类器）
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

        # 每5个epoch记录权重直方图、激活值分布、特征图、梯度范数
        if epoch % 5 == 0 or epoch == 1:
            log_weight_histograms(writer, model, epoch)
            log_layer_analysis(writer, model, sample_images, epoch)
            log_gradient_stats(writer, model, epoch)

        # 更新最佳准确率并保存最佳模型
        is_best = epoch_val_acc > best_acc
        if is_best:
            best_acc = epoch_val_acc
            best_checkpoint = {
                'epoch': epoch,
                'timestamp': timestamp,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_acc,
            }
            best_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(best_checkpoint, best_path)

        # 打印训练进度
        print(f" [{epoch:2d}/{EPOCHS}]   {epoch_train_loss:>10.4f}   {epoch_train_acc:>8.2f}%   "
              f"{epoch_val_loss:>8.4f}   {epoch_val_acc:>8.2f}%   {current_lr:>8.6f}   "
              f"{format_time(epoch_time):>8}   {samples_per_second:>8.0f}{'  *BEST*' if is_best else ''}")

    # -------------------- 保存最终模型 --------------------
    final_checkpoint = {
        'timestamp': timestamp,
        'epoch': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_acc,
    }
    final_path = os.path.join(SAVE_DIR, f"googlenet_{timestamp}_final_epoch{EPOCHS}_acc{best_acc:.1f}.pth")
    torch.save(final_checkpoint, final_path)

    # -------------------- 训练结束信息 --------------------
    print("\n" + "=" * 95)
    print(f"Training Complete  |  Best Val Acc: {best_acc:.2f}%")
    print(f"本次训练保存: {final_path}")
    print(f"最佳模型链接: {best_path}")
    print(f"TensorBoard 日志: {log_dir}")
    print(f"配置文件: {os.path.join(log_dir, 'config.yaml')}")
    print(f"查看可视化: tensorboard --logdir={log_dir}")
    print("=" * 95)

    writer.add_hparams(hparam_dict, {"hparam/best_val_acc": best_acc})
    writer.close()

    
# 工具函数

def collect_config():
    """自动收集模块级超参数，按类别分组返回"""
    config_groups = {}

    config_groups["训练参数"] = [
        ("Epochs", EPOCHS),
        ("Batch Size", BATCH_SIZE),
        ("Device", str(DEVICE)),
    ]

    config_groups["优化器"] = [
        ("Type", OPTIMIZER_TYPE),
        ("Learning Rate", LEARNING_RATE),
        ("Momentum", MOMENTUM),
        ("Weight Decay", WEIGHT_DECAY),
    ]

    config_groups["学习率调度"] = [
        ("Type", SCHEDULER_TYPE),
        ("T_max", SCHEDULER_T_MAX),
        ("eta_min", SCHEDULER_ETA_MIN),
    ]

    config_groups["模型结构"] = [
        ("Num Classes", NUM_CLASSES),
        ("Alpha", ALPHA),
        ("Aux Logits", AUX_LOGITS),
        ("Dropout", DROPOUT),
        ("Dropout Aux", DROPOUT_AUX),
        ("Aux Loss Weight", AUX_LOSS_WEIGHT),
    ]

    config_groups["数据"] = [
        ("Dataset", "CIFAR-10"),
        ("Normalize Mean", CIFAR10_MEAN),
        ("Normalize Std", CIFAR10_STD),
    ]

    return config_groups


def print_config(config_groups, model, log_dir):
    """结构化打印配置信息和模型统计"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 95)
    print(f"{'GoogLeNet Training on CIFAR-10':^70}")
    print("=" * 95)

    for group_name, items in config_groups.items():
        print(f"\n  [{group_name}]")
        for key, value in items:
            print(f"    {key:<18}: {value}")

    print(f"\n  [模型统计]")
    print(f"    {'Total Params':<18}: {total_params:,}")
    print(f"    {'Trainable Params':<18}: {trainable_params:,}")

    print(f"\n  [路径]")
    print(f"    {'TensorBoard':<18}: {log_dir}")
    print(f"    {'Checkpoint Dir':<18}: {SAVE_DIR}")

    print("\n" + "=" * 95)
    print(f"  TensorBoard: tensorboard --logdir={log_dir}")
    print("=" * 95 + "\n")


def log_weight_histograms(writer, model, epoch):
    """
    记录模型权重和梯度的直方图到 TensorBoard

    用于诊断:
    - 梯度消失: 梯度分布集中在 0 附近
    - 梯度爆炸: 梯度分布范围极大
    - 权重初始化问题: 权重分布不合理
    """
    for name, param in model.named_parameters():
        if not param.requires_grad or param.numel() == 0:
            continue
        w = param.data.detach().cpu().float()
        if w.isfinite().all():
            writer.add_histogram(f"Weights/{name}", w, epoch, max_bins=64)
        if param.grad is not None and param.grad.numel() > 0:
            g = param.grad.detach().cpu().float()
            if g.isfinite().all():
                writer.add_histogram(f"Gradients/{name}", g, epoch, max_bins=64)


def build_model_summary_text(model):
    """构建模型结构文本摘要，记录到 TensorBoard TEXT 标签页"""
    lines = []
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_counts = {"Conv2d": 0, "Linear": 0, "MaxPool2d": 0, "AdaptiveAvgPool2d": 0, "Inception": 0}
    for m in model.modules():
        cls = m.__class__.__name__
        if cls in layer_counts:
            layer_counts[cls] += 1

    lines.append(f"GoogLeNet on CIFAR-10")
    lines.append(f"{'='*50}")
    lines.append(f"Total params:     {total_params:>12,}")
    lines.append(f"Trainable params: {trainable_params:>12,}")
    lines.append(f"{'='*50}")
    lines.append(f"Layers breakdown:")
    for name, count in layer_counts.items():
        if count > 0:
            lines.append(f"  {name:<20}: {count}")
    lines.append(f"{'='*50}")
    lines.append(f"")
    for name, module in model.named_children():
        lines.append(f"  ({name}): {module.__class__.__name__}")
    return "\n".join(lines)


def log_layer_analysis(writer, model, sample_images, epoch):
    """
    一次前向传播同时完成激活值分布和特征图记录
    在 hook 内即时写入 histogram 以减少内存峰值
    """
    feature_maps = {}

    def hook_fn(module, input, output, name):
        if isinstance(output, torch.Tensor):
            act_cpu = output.detach().cpu().float()
            if act_cpu.isfinite().all():
                writer.add_histogram(f"Activations/{name}", act_cpu, epoch, max_bins=64)
            if output.dim() == 4:
                feature_maps[name] = act_cpu

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            hooks.append(module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(sample_images[:1].to(DEVICE))
    if was_training:
        model.train()

    for i, (name, fm) in enumerate(feature_maps.items()):
        if i >= 6:
            break
        n_channels = min(8, fm.size(1))
        grid = fm[0, :n_channels].unsqueeze(1)
        writer.add_images(f"FeatureMaps/{name}", grid, epoch, dataformats="NCHW")

    for h in hooks:
        h.remove()


def log_gradient_stats(writer, model, epoch):
    """记录各层梯度范数和总梯度范数到 TensorBoard"""
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            writer.add_scalar(f"GradientNorm/{name}", param_norm.item(), epoch)
    total_norm = total_norm ** 0.5
    writer.add_scalar("GradientNorm/total", total_norm, epoch)
    return total_norm


def describe_transforms(transform):
    """从 transforms.Compose 对象中提取参数，返回结构化描述"""
    import torchvision.transforms as T

    def _l(v):
        """tuple/list → list，标量原样返回，确保 yaml 无 !!python/tuple"""
        if isinstance(v, tuple):
            return [round(x, 4) if isinstance(x, float) else x for x in v]
        if isinstance(v, list):
            return [round(x, 4) if isinstance(x, float) else x for x in v]
        return v

    if not isinstance(transform, transforms.Compose):
        return str(transform)

    steps = []
    for t in transform.transforms:
        cls_name = t.__class__.__name__
        if isinstance(t, T.Normalize):
            steps.append({cls_name: {"mean": _l(t.mean), "std": _l(t.std)}})
        elif isinstance(t, T.RandomHorizontalFlip):
            steps.append({cls_name: {"p": t.p}})
        elif isinstance(t, T.RandomVerticalFlip):
            steps.append({cls_name: {"p": t.p}})
        elif isinstance(t, T.RandomRotation):
            steps.append({cls_name: {"degrees": _l(t.degrees)}})
        elif isinstance(t, T.RandomCrop):
            steps.append({cls_name: {"size": _l(t.size), "padding": t.padding}})
        elif isinstance(t, T.RandomResizedCrop):
            steps.append({cls_name: {"size": _l(t.size), "scale": _l(t.scale), "ratio": _l(t.ratio)}})
        elif isinstance(t, T.ColorJitter):
            steps.append({cls_name: {"brightness": _l(t.brightness), "contrast": _l(t.contrast),
                                     "saturation": _l(t.saturation), "hue": _l(t.hue)}})
        elif isinstance(t, (T.Resize, T.CenterCrop)):
            steps.append({cls_name: {"size": _l(t.size)}})
        elif isinstance(t, T.Grayscale):
            steps.append({cls_name: {"num_output_channels": t.num_output_channels}})
        elif isinstance(t, T.RandomErasing):
            steps.append({cls_name: {"p": t.p, "scale": _l(t.scale), "ratio": _l(t.ratio)}})
        else:
            steps.append(cls_name)
    return steps


def save_config_yaml(log_dir, model, **kwargs):
    """生成可复现训练的 config.yaml，扁平结构，无冗余"""

    # 统计各层数量（信息性，非参数）
    layer_counts = {}
    for m in model.modules():
        cls = m.__class__.__name__
        if cls not in ("BasicConv2d", "GoogLeNet", "Sequential"):
            layer_counts[cls] = layer_counts.get(cls, 0) + 1

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = {
        "model": {
            "name": "GoogLeNet",
            "num_classes": NUM_CLASSES,
            "alpha": ALPHA,
            "dropout": DROPOUT,
            "dropout_aux": DROPOUT_AUX,
            "aux_logits": AUX_LOGITS,
        },

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),

        "optimizer": OPTIMIZER_TYPE,
        "lr": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,

        "scheduler": SCHEDULER_TYPE,
        "T_max": SCHEDULER_T_MAX,
        "eta_min": SCHEDULER_ETA_MIN,

        "criterion": "CrossEntropyLoss",
        "aux_loss_weight": AUX_LOSS_WEIGHT,

        "dataset": "CIFAR-10",
        "train_samples": kwargs.get("train_samples", 50000),
        "val_samples": kwargs.get("val_samples", 10000),

        "augmentation": {
            "train": describe_transforms(kwargs.get("transform_train")),
            "val": describe_transforms(kwargs.get("transform_val")),
        },

        "stats": {
            "total_params": total_p,
            "trainable_params": trainable_p,
            "layers": layer_counts,
            "model_size_mb": round(total_p * 4 / (1024 ** 2), 2),
        },

        "timestamp": kwargs.get("timestamp", ""),
    }

    config_path = os.path.join(log_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"配置文件已保存: {config_path}")


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
