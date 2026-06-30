"""
VGG-11 模型实现
"""

import torch
import torch.nn as nn
from typing import List, Union, cast

__all__ = ["VGG", "vgg11"]


# VGG-11 配置 (配置 A)
# 数字表示输出通道数，"M" 表示 MaxPool
VGG11_CFG: List[Union[str, int]] = [
    32, #"M",          #64, "M", 
    64, "M",         #128, "M", 
    128, 128, "M",    #256, 256, "M", 
    256, 256, "M",    #512, 512, "M",   
    512, 512#, "M"     #512, 512, "M"
]


def make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    """
    根据配置构建 VGG 的特征提取层
    
    Args:
        cfg: 层配置列表
        
    Returns:
        nn.Sequential 特征提取层
    """
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    VGG 网络模型
    
    Args:
        features: 特征提取层
        num_classes: 分类数量
        init_weights: 是否初始化权重
        dropout: Dropout 概率
    """
    
    def __init__(
        self, 
        features: nn.Module, 
        num_classes: int = 1000, 
        init_weights: bool = True, 
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(

            #nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(512 * 4 * 4, 4096),

            nn.ReLU(True),
            nn.Dropout(p=dropout),

            #nn.Linear(4096, 4096),
            nn.Linear(4096, 512),

            nn.ReLU(True),
            nn.Dropout(p=dropout),

            #nn.Linear(4096, num_classes),
            nn.Linear(512, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """初始化权重 (官方实现方式)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg11(num_classes: int = 1000, dropout: float = 0.5) -> VGG:
    """
    VGG-11 模型
    
    Args:
        num_classes: 分类数量
        dropout: Dropout 概率
        
    Returns:
        VGG-11 模型
    """
    return VGG(make_layers(VGG11_CFG), num_classes=num_classes, dropout=dropout)


def print_layer_shapes(model: nn.Module, input_size: tuple = (128, 3, 32, 32)):
    """打印模型每一层的输出形状"""
    x = torch.randn(*input_size)
    print(f"\n{'Layer':<40} {'Output Shape':<25}")
    print("-" * 65)
    print(f"{'Input':<40} {str(list(x.shape)):<25}")
    print("-" * 65)

    def hook_fn(module, input, output, name):
        if isinstance(output, torch.Tensor):
            print(f"{name:<40} {str(list(output.shape)):<25}")
        return None

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.Dropout)):
            hook = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            hooks.append(hook)

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    model = vgg11(num_classes=10)
    print(model)
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
    print_layer_shapes(model)
