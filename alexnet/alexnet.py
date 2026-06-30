import torch
import torch.nn as nn

__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),

            nn.ReLU(inplace=True),

            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),

            nn.ReLU(inplace=True),

            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

            #nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),

            #nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),

            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),

            #nn.Linear(256 * 6 * 6, 4096),
            nn.Linear(256 * 3 * 3, 1536),

            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            #nn.Linear(4096, 4096),
            nn.Linear(1536, 512),

            nn.ReLU(inplace=True),

            #nn.Linear(4096, num_classes),
            nn.Linear(512, num_classes),
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(num_classes: int = 1000, dropout: float = 0.5) -> AlexNet:
    """构建 AlexNet 模型（随机初始化）"""
    return AlexNet(num_classes=num_classes, dropout=dropout)


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
    model = alexnet(num_classes=10)
    print(model)
    print_layer_shapes(model)