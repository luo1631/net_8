# 数据集介绍

本目录包含项目使用的所有数据集。

---

## 1. CIFAR-10

**路径**: `cifar-10-batches-py/`

### 简介
CIFAR-10 是一个经典的图像分类数据集，广泛用于机器学习和计算机视觉研究。

### 数据规格

| 属性 | 值 |
|------|-----|
| 图像尺寸 | 32 × 32 像素 |
| 颜色通道 | 3 (RGB) |
| 训练样本数 | 50,000 |
| 测试样本数 | 10,000 |
| 类别数 | 10 |

### 类别列表
```
0: airplane (飞机)
1: automobile (汽车)
2: bird (鸟)
3: cat (猫)
4: deer (鹿)
5: dog (狗)
6: frog (青蛙)
7: horse (马)
8: ship (船)
9: truck (卡车)
```

### 文件结构
```
cifar-10-batches-py/
├── data_batch_1    # 训练数据批次 1 (10000张)
├── data_batch_2    # 训练数据批次 2 (10000张)
├── data_batch_3    # 训练数据批次 3 (10000张)
├── data_batch_4    # 训练数据批次 4 (10000张)
├── data_batch_5    # 训练数据批次 5 (10000张)
├── test_batch      # 测试数据 (10000张)
├── batches.meta    # 元数据
└── readme.html     # 说明文档
```

### PyTorch 使用示例
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),  # AlexNet 需要 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

---

## 2. FashionMNIST

**路径**: `FashionMNIST/raw/`

### 简介
FashionMNIST 是 MNIST 的现代替代品，包含服装图像，比手写数字更具挑战性。

### 数据规格

| 属性 | 值 |
|------|-----|
| 图像尺寸 | 28 × 28 像素 |
| 颜色通道 | 1 (灰度) |
| 训练样本数 | 60,000 |
| 测试样本数 | 10,000 |
| 类别数 | 10 |

### 类别列表
```
0: T-shirt/top (T恤)
1: Trouser (裤子)
2: Pullover (套头衫)
3: Dress (连衣裙)
4: Coat (外套)
5: Sandal (凉鞋)
6: Shirt (衬衫)
7: Sneaker (运动鞋)
8: Bag (包)
9: Ankle boot (短靴)
```

### 文件结构
```
FashionMNIST/raw/
├── train-images-idx3-ubyte     # 训练图像
├── train-labels-idx1-ubyte     # 训练标签
├── t10k-images-idx3-ubyte      # 测试图像
└── t10k-labels-idx1-ubyte      # 测试标签
```

### PyTorch 使用示例
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
```

---

## 3. fra-eng (法语-英语翻译)

**路径**: `fra-eng/`

### 简介
这是一个法语到英语的机器翻译数据集，来源于 Tatoeba 项目。

### 数据规格

| 属性 | 值 |
|------|-----|
| 文件 | `fra.txt` |
| 格式 | `英语句子\t法语句子` |
| 句子对数量 | 约 190,000+ |

### 文件结构
```
fra-eng/
├── fra.txt        # 主数据文件 (英语\t法语)
└── _about.txt     # 数据集说明
```

### 数据格式示例
```
Go.	Va !
Run!	Cours !
Run!	Courez !
Who?	Qui ?
Wow!	Ça alors !
```

### PyTorch 使用示例
```python
# 读取翻译数据
pairs = []
with open('./data/fra-eng/fra.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))  # (英语, 法语)

print(f"共 {len(pairs)} 个句子对")
```

---

## 数据集对比

| 数据集 | 类型 | 输入尺寸 | 样本数 | 任务 |
|--------|------|----------|--------|------|
| CIFAR-10 | 图像 | 32×32×3 | 60,000 | 10类分类 |
| FashionMNIST | 图像 | 28×28×1 | 70,000 | 10类分类 |
| fra-eng | 文本 | 句子对 | ~190,000 | 机器翻译 |

---

## 推荐模型

| 数据集 | 推荐模型 |
|--------|----------|
| CIFAR-10 | AlexNet, ResNet, VGG |
| FashionMNIST | LeNet, MLP, CNN |
| fra-eng | Transformer, Seq2Seq, RNN |
