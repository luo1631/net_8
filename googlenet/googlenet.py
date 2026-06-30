"""
GoogLeNet (Inception v1) 模型实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from typing import Optional

__all__ = ["GoogLeNet", "googlenet"]


GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {
    "logits": Tensor,
    "aux_logits2": Optional[Tensor],
    "aux_logits1": Optional[Tensor],
}


def _make_divisible(v: int, divisor: int = 8) -> int:
    """确保通道数能被 divisor 整除"""
    return max(divisor, (v + divisor // 2) // divisor * divisor)


class BasicConv2d(nn.Module):
    """卷积 + BatchNorm + ReLU"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    """
    Inception 模块 (针对 CIFAR-10 修改 —— 所有分支保持空间尺寸不变)

    4 个分支并行计算后在通道维度拼接:
    - branch1: 1x1 卷积
    - branch2: 1x1 卷积 → 3x3 卷积
    - branch3: 1x1 卷积 → 5x5 卷积
    - branch4: 3x3 MaxPool → 1x1 卷积

    Args:
        in_channels: 输入通道数
        ch1x1: branch1 输出通道数
        ch3x3red: branch2 1x1 降维通道数
        ch3x3: branch2 3x3 输出通道数
        ch5x5red: branch3 1x1 降维通道数
        ch5x5: branch3 5x5 输出通道数
        pool_proj: branch4 1x1 输出通道数
    """
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )


    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionAux(nn.Module):
    """
    辅助分类器
    
    训练时提供额外的梯度信号，帮助缓解梯度消失问题
    
    Args:
        in_channels: 输入通道数
        num_classes: 分类数量
        hidden_channels: 隐藏层通道数
        dropout: Dropout 概率
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        fc_hidden: int = 1024,
        dropout: float = 0.7
    ) -> None:
        super().__init__()
        self.conv = BasicConv2d(in_channels, hidden_channels, kernel_size=1)
        self.fc1 = nn.Linear(hidden_channels * 4 * 4, fc_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) 网络模型
    
    Args:
        num_classes: 分类数量
        aux_logits: 是否使用辅助分类器
        alpha: 通道数缩放比例 (默认 1.0)
        dropout: 主分类器 Dropout 概率
        dropout_aux: 辅助分类器 Dropout 概率
        init_weights: 是否初始化权重
    """

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        alpha: float = 1.0,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.aux_logits = aux_logits
        self.alpha = alpha

        def _ch(ch: int) -> int:
            return _make_divisible(int(ch * alpha))

        # -------------------- 前置卷积层 --------------------
        #self.conv1 = BasicConv2d(3, _ch(64), kernel_size=7, stride=2, padding=3)
        self.conv1 = BasicConv2d(3, _ch(64), kernel_size=3, stride=1, padding=1)
        #self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(_ch(64), _ch(64), kernel_size=1)
        self.conv3 = BasicConv2d(_ch(64), _ch(192), kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # -------------------- Inception 模块 --------------------
        # Inception(in_c, 1x1, 3red, 3x3, 5red, 5x5, pool)
        self.inception3a = Inception(_ch(192), _ch(64), _ch(96), _ch(128), _ch(16), _ch(32), _ch(32))
        self.inception3b = Inception(_ch(256), _ch(128), _ch(128), _ch(192), _ch(32), _ch(96), _ch(64))
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(_ch(480), _ch(192), _ch(96), _ch(208), _ch(16), _ch(48), _ch(64))
        self.aux1 = InceptionAux(_ch(512), num_classes, _ch(128), _ch(1024), dropout=dropout_aux) if aux_logits else None

        self.inception4b = Inception(_ch(512), _ch(160), _ch(112), _ch(224), _ch(24), _ch(64), _ch(64))
        self.inception4c = Inception(_ch(512), _ch(128), _ch(128), _ch(256), _ch(24), _ch(64), _ch(64))
        self.inception4d = Inception(_ch(512), _ch(112), _ch(144), _ch(288), _ch(32), _ch(64), _ch(64))
        self.aux2 = InceptionAux(_ch(528), num_classes, _ch(128), _ch(1024), dropout=dropout_aux) if aux_logits else None

        self.inception4e = Inception(_ch(528), _ch(256), _ch(160), _ch(320), _ch(32), _ch(128), _ch(128))
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(_ch(832), _ch(256), _ch(160), _ch(320), _ch(32), _ch(128), _ch(128))
        self.inception5b = Inception(_ch(832), _ch(384), _ch(192), _ch(384), _ch(48), _ch(128), _ch(128))

        # -------------------- 主分类器 --------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(_ch(1024), num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: Tensor):
        x = self._forward(x)
        if self.training and self.aux_logits:
            return GoogLeNetOutputs(x[0], x[1], x[2])
        return x[0]

    def _forward(self, x: Tensor) -> tuple:
        x = self.conv1(x)
        #x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training and self.aux1 is not None else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.training and self.aux2 is not None else None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux2, aux1
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                nn.init.constant_(m.bias, 0)

def googlenet(
    num_classes: int = 1000, 
    aux_logits: bool = True, 
    alpha: float = 1.0,
    dropout: float = 0.2,
    dropout_aux: float = 0.7
) -> GoogLeNet:
    """
    构建 GoogLeNet 模型
    
    Args:
        num_classes: 分类数量
        aux_logits: 是否使用辅助分类器
        alpha: 通道数缩放比例 (默认 1.0)
        dropout: 主分类器 Dropout 概率
        dropout_aux: 辅助分类器 Dropout 概率

    Returns:
        GoogLeNet 模型
    """
    return GoogLeNet(
        num_classes=num_classes, 
        aux_logits=aux_logits, 
        alpha=alpha, 
        dropout=dropout,
        dropout_aux=dropout_aux
    )


def save_layer_shapes(model: nn.Module, filepath: str, input_size: tuple = (128, 3, 32, 32)):
    """将模型每一层的输出形状写入文件（训练模式以包含辅助分类器）"""
    model.train()
    x = torch.randn(*input_size)
    lines = []
    lines.append(f"\n{'Layer':<40} {'Output Shape':<25}")
    lines.append("-" * 65)
    lines.append(f"{'Input':<40} {str(list(x.shape)):<25}")
    lines.append("-" * 65)

    def hook_fn(module, input, output, name):
        if isinstance(output, torch.Tensor):
            lines.append(f"{name:<40} {str(list(output.shape)):<25}")

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout, nn.BatchNorm2d, nn.ReLU)):
            hook = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            hooks.append(hook)

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def print_model_summary(model: nn.Module):
    """紧凑打印模型顶层结构，避免递归展开导致的终端输出截断"""
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print(f"  GoogLeNet  (alpha={model.alpha}, num_classes={model.fc.out_features})")
    print(f"  Total params: {total_p:>12,}   Trainable: {trainable_p:>12,}")
    print(f"{'='*70}")
    print(f"  {'Module':<22} {'Type':<18} {'Details'}")
    print(f"  {'-'*60}")

    for name, module in model.named_children():
        cls = module.__class__.__name__
        if cls == "BasicConv2d":
            in_c = module.conv.in_channels
            out_c = module.conv.out_channels
            k = module.conv.kernel_size
            s = module.conv.stride
            detail = f"{in_c}->{out_c}, k={k}, s={s}"
        elif cls == "Inception":
            in_c = module.branch1.conv.in_channels
            out_c = (module.branch1.conv.out_channels + module.branch2[1].conv.out_channels
                     + module.branch3[1].conv.out_channels + module.branch4[1].conv.out_channels)
            detail = f"{in_c}->{out_c}"
        elif cls == "MaxPool2d":
            detail = f"k={module.kernel_size}, s={module.stride}"
        elif cls == "InceptionAux":
            in_c = module.conv.conv.in_channels
            detail = f"in={in_c}"
        elif cls == "AdaptiveAvgPool2d":
            detail = f"output={module.output_size}"
        elif cls == "Dropout":
            detail = f"p={module.p}"
        elif cls == "Linear":
            detail = f"{module.in_features}->{module.out_features}"
        else:
            detail = ""
        print(f"  {name:<22} {cls:<18} {detail}")
    print(f"{'='*70}\n")


def save_model_yaml(model: nn.Module, filepath: str):
    """将完整模型架构及参数信息写入文件"""
    total_p = sum(p.numel() for p in model.parameters())
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(model))
        f.write(f"\n\n参数量: {total_p:,}\n")


if __name__ == "__main__":
    model = googlenet(num_classes=10, alpha=0.5)
    print_model_summary(model)

    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.yaml")
    save_model_yaml(model, yaml_path)
    save_layer_shapes(model, yaml_path) 
    print(f"完整架构+层级形状已保存: {yaml_path}")
