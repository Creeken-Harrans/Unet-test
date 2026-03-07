# U-Net 语义分割代码仓库说明

## 1. 项目概览

这是一个基于 **PyTorch + U-Net** 的二值语义分割练习仓库。仓库根目录下的核心脚本是：

- `train.py`：训练入口
- `dataset.py`：数据集读取与增强
- `unet_model.py`：U-Net 整体结构
- `unet_parts.py`：U-Net 各个基础模块
- `predict.py`：推理与结果保存
- `best_model.pth`：当前仓库内已有的一份模型权重

根目录代码本质上是对 `Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/` 教学示例的一份抽取与改写：
- **模型代码**放在了仓库根目录；
- **训练/测试数据**仍直接引用嵌套目录中的 `lesson-2/data/`；
- 因此，这个仓库的实际可运行主线，是“**根目录脚本 + 嵌套目录中的示例数据**”。

---

## 2. 当前仓库的实际数据组织

训练脚本默认使用的数据路径是：

```text
Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/train
```

对应目录结构如下：

```text
Deep-Learning-master/
└── Deep-Learning-master/
    └── Pytorch-Seg/
        └── lesson-2/
            └── data/
                ├── train/
                │   ├── image/
                │   └── label/
                └── test/
```

通过仓库检查可知：

- `train/image`：30 张图片
- `train/label`：30 张标签
- `test`：30 张测试图片
- 示例图片分辨率：`512 x 512`
- 图片类型：8-bit 灰度 PNG

该项目是一个**单通道输入、单通道输出**的二分类分割任务：
- 输入：灰度图
- 标签：黑白掩码
- 输出：每个像素属于前景的概率（训练时是 logits，推理时经 sigmoid 转成概率）

---

## 3. 仓库结构与职责划分

```text
.
├── train.py          # 训练入口
├── dataset.py        # 数据读取、标签构造、简单增强
├── unet_model.py     # U-Net 主体网络
├── unet_parts.py     # DoubleConv / Down / Up / OutConv
├── predict.py        # 推理脚本，输出 result/*.png
├── best_model.pth    # 模型权重
└── Deep-Learning-master/
    └── .../lesson-2/data/
        ├── train/image/*.png
        ├── train/label/*.png
        └── test/*.png
```

如果只看“真正参与主流程的代码”，重点只需要读这 5 个根目录文件：

1. `train.py`
2. `dataset.py`
3. `unet_model.py`
4. `unet_parts.py`
5. `predict.py`

---

## 4. 从 `train.py` 出发的完整训练流程

这一部分是整个仓库最核心的代码阅读路径。

### 4.1 入口逻辑

`train.py` 的主程序做了 4 件事：

1. 选择运行设备：优先 `cuda`，否则 `cpu`
2. 创建模型：`UNet(n_channels=1, n_classes=1)`
3. 把模型搬到设备上：`net.to(device)`
4. 指定训练数据路径并调用 `train_net(...)`

也就是说，训练主线可以概括成：

```text
main
 -> 选择 device
 -> 初始化 UNet
 -> 指定 data_path
 -> 调用 train_net
```

这里的参数设置说明：

- `n_channels=1`：输入是灰度图
- `n_classes=1`：输出 1 个通道，表示前景类别的分割结果

由于损失函数使用的是 `BCEWithLogitsLoss`，所以网络最后输出的是 **logits**，而不是已经经过 sigmoid 的概率图。

---

### 4.2 `train_net()` 的执行流程

`train_net(net, device, data_path, epochs=40, batch_size=1, lr=1e-5)` 是训练的主函数，执行顺序如下：

#### 第一步：构造数据集对象

```python
isbi_dataset = ISBI_Loader(data_path)
```

这里进入 `dataset.py`，把 `data_path/image/*.png` 下的图片都扫出来，形成样本列表。

#### 第二步：构造 DataLoader

```python
train_loader = torch.utils.data.DataLoader(
    dataset=isbi_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

作用：
- 按 batch 读取样本
- 每个 epoch 打乱顺序
- 当前默认 `batch_size=1`

#### 第三步：定义优化器

```python
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
```

当前训练采用：
- 优化器：`RMSprop`
- 学习率：`1e-5`
- `weight_decay=1e-8`
- `momentum=0.9`

#### 第四步：定义损失函数

```python
criterion = nn.BCEWithLogitsLoss()
```

这一步非常关键，它隐含了两层含义：

1. 当前任务是**二值分割**；
2. 模型输出不能提前做 sigmoid，因为 `BCEWithLogitsLoss` 内部已经集成了 sigmoid 的数值稳定实现。

所以训练阶段流程是：

```text
模型输出 logits -> BCEWithLogitsLoss(logits, label)
```

而不是：

```text
模型输出 sigmoid 后的概率 -> 再送入 BCEWithLogitsLoss
```

#### 第五步：初始化最优损失

```python
best_loss = float('inf')
```

用来记录训练过程中出现过的最小 loss，并据此保存权重。

#### 第六步：进入 epoch 循环

```python
for epoch in range(epochs):
    net.train()
    for image, label in train_loader:
        ...
```

外层循环控制训练轮数，默认：
- `epochs = 40`

内层循环逐 batch 取数据。

#### 第七步：单个 batch 的训练细节

每个 batch 内部的执行顺序如下：

1. `optimizer.zero_grad()`：清空旧梯度
2. `image.to(device, dtype=torch.float32)`：图像转到设备并转成 `float32`
3. `label.to(device, dtype=torch.float32)`：标签转到设备并转成 `float32`
4. `pred = net(image)`：前向传播，得到 logits
5. `loss = criterion(pred, label)`：计算损失
6. `print("Loss/train", loss.item())`：打印训练损失
7. 如果当前 loss 更小，则保存 `best_model.pth`
8. `loss.backward()`：反向传播
9. `optimizer.step()`：更新参数

可以写成一条清晰的数据流：

```text
image,label
 -> 搬到 device
 -> net(image)
 -> pred(logits)
 -> criterion(pred, label)
 -> loss
 -> backward
 -> optimizer.step
```

---

## 5. `dataset.py`：数据是如何被读进来的

`dataset.py` 中的 `ISBI_Loader` 负责整个数据读取过程。

### 5.1 初始化阶段：收集所有训练图片路径

在 `__init__` 中，程序会拼出搜索路径：

```text
data_path/image/*.png
```

然后使用 `glob.glob(...)` 获取所有训练图片。

这意味着该数据集类默认假设你的目录结构必须是：

```text
train/
├── image/
└── label/
```

并且 `image` 和 `label` 中的文件名需要一一对应，例如：

```text
image/0.png  <->  label/0.png
image/1.png  <->  label/1.png
```

### 5.2 `__getitem__`：单个样本的读取流程

每次取样本时，执行流程如下：

#### 1）根据索引拿到图片路径

```python
image_path = self.imgs_path[index]
```

#### 2）通过字符串替换得到标签路径

```python
label_path = image_path.replace("image", "label")
```

这是一个非常“轻量”的配对方式，前提是路径中 `image` 和 `label` 的命名严格规范。

#### 3）使用 OpenCV 读取原图和标签图

```python
image = cv2.imread(image_path)
label = cv2.imread(label_path)
```

如果文件读不到，代码会直接抛出异常。

#### 4）转为灰度图

```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
```

于是单张图从三通道 BGR 变成二维灰度图。

#### 5）改成 `[C, H, W]` 格式

```python
image = image.reshape(1, H, W)
label = label.reshape(1, H, W)
```

这样就把单通道图像显式改成了 PyTorch 常用的通道优先格式。

#### 6）标签归一化

```python
if label.max() > 1:
    label = label / 255
```

这一句说明标签图原始像素值预计是：
- 背景：`0`
- 前景：`255`

除以 `255` 后，标签就变成：
- 背景：`0`
- 前景：`1`

这正好符合二分类分割中 `BCEWithLogitsLoss` 的标签要求。

#### 7）随机翻转增强

```python
flipCode = random.choice([-1, 0, 1, 2])
if flipCode != 2:
    image = self.augment(image, flipCode)
    label = self.augment(label, flipCode)
```

增强策略很简单：
- `1`：水平翻转
- `0`：垂直翻转
- `-1`：水平 + 垂直翻转
- `2`：不做增强

#### 8）返回图像和标签

```python
return image, label
```

配合 DataLoader 后，最终 batch 张量形状大致为：

```text
image: [B, 1, H, W]
label: [B, 1, H, W]
```

---

## 6. `unet_model.py` 与 `unet_parts.py`：模型结构拆解

### 6.1 U-Net 的模块组成

根目录中的 U-Net 由这几个模块拼装而成：

- `DoubleConv`：两层 `Conv + BN + ReLU`
- `Down`：`MaxPool` 下采样后接 `DoubleConv`
- `Up`：上采样 + skip connection 拼接 + `DoubleConv`
- `OutConv`：`1x1 Conv` 输出最终通道数

### 6.2 网络结构主干

`UNet` 的 forward 过程如下：

```text
x1 = inc(x)
x2 = down1(x1)
x3 = down2(x2)
x4 = down3(x3)
x5 = down4(x4)

x = up1(x5, x4)
x = up2(x, x3)
x = up3(x, x2)
x = up4(x, x1)

logits = outc(x)
```

这就是典型的 U 形结构：

```text
输入
 -> 编码器逐步下采样
 -> bottleneck
 -> 解码器逐步上采样
 -> 与编码器同层特征拼接
 -> 输出分割图
```

### 6.3 以 `512 x 512` 输入为例的尺寸变化

如果输入图片尺寸是 `[B, 1, 512, 512]`，那么大致会经历如下变化：

```text
输入    : [B,   1, 512, 512]
inc     : [B,  64, 512, 512]
down1   : [B, 128, 256, 256]
down2   : [B, 256, 128, 128]
down3   : [B, 512,  64,  64]
down4   : [B, 512,  32,  32]
up1     : [B, 256,  64,  64]
up2     : [B, 128, 128, 128]
up3     : [B,  64, 256, 256]
up4     : [B,  64, 512, 512]
outc    : [B,   1, 512, 512]
```

### 6.4 `Up` 模块中为什么要 `pad`

在上采样时，解码器特征图尺寸可能与编码器跳连特征图略有差异，所以 `Up.forward()` 里先计算高宽差值：

```python
diffY = x2.size(2) - x1.size(2)
diffX = x2.size(3) - x1.size(3)
```

然后用 `F.pad(...)` 把上采样结果补齐，最后再做拼接：

```python
x = torch.cat([x2, x1], dim=1)
```

这一步是 U-Net 中 skip connection 可以顺利拼接的关键。

---

## 7. `predict.py`：推理流程

推理脚本和训练脚本的设计思想一致，但流程更简单：

### 7.1 主要步骤

1. 选择 `cuda/cpu`
2. 初始化 `UNet(n_channels=1, n_classes=1)`
3. 加载 `best_model.pth`
4. 切换到 `eval()` 模式
5. 扫描测试目录下所有 `*.png`
6. 逐张读取图片并转成灰度图
7. reshape 成 `[1, 1, H, W]`
8. 前向推理得到 logits
9. 手动做 `sigmoid`
10. 用 `0.5` 阈值二值化
11. 输出到 `result/*_res.png`

### 7.2 为什么预测时要手动 `sigmoid`

训练时使用的是 `BCEWithLogitsLoss`，因此模型本身输出的是 logits。

所以预测阶段必须手动执行：

```python
pred = torch.sigmoid(pred)
```

这样才能把网络输出映射到 `[0, 1]` 概率范围内。随后再按阈值 `0.5` 切成二值掩码。

### 7.3 推理结果保存规则

如果输入是：

```text
xxx/test/17.png
```

那么输出会保存为：

```text
result/17_res.png
```

`result/` 目录不存在时会自动创建。

---

## 8. 一张图看懂整体主线

```text
train.py
 -> 初始化 UNet
 -> 构造 ISBI_Loader
 -> DataLoader 按 batch 取数据
 -> 前向传播得到 logits
 -> BCEWithLogitsLoss 计算损失
 -> backward()
 -> optimizer.step()
 -> 保存 best_model.pth

predict.py
 -> 加载 best_model.pth
 -> 读取 test/*.png
 -> UNet 前向传播
 -> sigmoid
 -> 0.5 阈值化
 -> 保存到 result/*.png
```

---

## 9. 运行方式

### 9.1 依赖

至少需要以下 Python 包：

```bash
pip install torch numpy opencv-python
```

如果你使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch。

### 9.2 训练

在仓库根目录执行：

```bash
python train.py
```

默认会：
- 训练 40 个 epoch
- 使用 batch size = 1
- 学习率 `1e-5`
- 将最优权重保存为根目录下的 `best_model.pth`

### 9.3 推理

在仓库根目录执行：

```bash
python predict.py
```

推理结果默认保存在：

```text
result/
```

---

## 10. 代码阅读后的关键结论

这个仓库的代码主线非常清晰，本质上是一套最小可运行的 U-Net 二值分割示例：

- `train.py` 负责训练调度
- `dataset.py` 负责样本读取与标签处理
- `unet_model.py` / `unet_parts.py` 负责网络结构
- `predict.py` 负责离线推理

如果你是从学习角度阅读，这个项目非常适合用来理解下面这些核心概念：

1. **PyTorch 训练循环的最小闭环**
2. **自定义 Dataset 与 DataLoader 的配合方式**
3. **U-Net 编码器 / 解码器 / skip connection 的实现方式**
4. **`BCEWithLogitsLoss` 与 sigmoid 在训练/推理阶段的职责划分**
5. **二值分割结果从 logits 到掩码图的完整落地流程**

---

## 11. 当前实现中的注意事项与潜在问题

这部分不是“看不懂代码”的问题，而是“读懂以后会发现的实现细节”。

### 11.1 `predict.py` 的测试路径是 Windows 风格写法

`predict.py` 里写的是：

```python
test_dir = r"Deep-Learning-master\Deep-Learning-master\Pytorch-Seg\lesson-2\data\test"
```

这在 Windows 上通常没问题，但在 Linux / macOS 环境下，反斜杠会被当作普通字符，可能导致找不到图片。

更稳妥的写法应该是使用：
- 正斜杠路径，或
- `os.path.join(...)`

### 11.2 数据增强的位置有一个容易忽略的细节

当前代码是：

1. 先把图像 reshape 成 `[1, H, W]`
2. 再调用 `cv2.flip(...)`

但 OpenCV 更自然处理的是：
- 二维灰度图：`[H, W]`
- 或三维彩色图：`[H, W, C]`

因此，对 `[1, H, W]` 调用 `cv2.flip(...)` 并不是最标准的写法，增强语义可能与预期的“按高宽翻转”不完全一致。

更合理的方式通常是：
- 先在 `[H, W]` 上做翻转，最后再 reshape 成 `[1, H, W]`；
- 或改用 `numpy` / `torch` 按明确维度翻转。

### 11.3 `best_loss` 更适合保存成 Python 标量

当前代码使用：

```python
if loss < best_loss:
    best_loss = loss
```

这里的 `loss` 是张量。虽然在很多情况下这段代码能工作，但从可读性和稳定性上说，更推荐使用：

```python
loss_value = loss.item()
```

再用 `loss_value` 去比较和保存。

### 11.4 当前训练过程没有验证集

代码里保存“最佳模型”的标准是：
- **训练过程中的最小 batch loss**

而不是：
- 验证集 loss
- 验证集 Dice / IoU

因此，这里的“best model”只是“训练 batch 上最小损失对应的权重”，不等价于泛化能力最好的模型。

### 11.5 输入图像没有显式归一化到 `[0, 1]`

当前代码中：
- 标签做了 `0/255 -> 0/1`
- 输入图像没有做 `/255`

也就是说，图像像素以 `0~255` 的灰度值直接送入网络。

这不一定完全不能训练，但通常更常见的做法是把输入也归一化到 `[0, 1]`，这样数值范围更稳定。

---

## 12. 总结

如果一句话总结这个仓库：

> 这是一个围绕 `train.py` 组织起来的、结构清楚、适合教学入门的 U-Net 二值分割最小实现。

它的优点是：
- 代码短
- 主线清楚
- 容易追踪数据流
- 很适合学习 PyTorch 分割项目的基本结构

它的局限也很明显：
- 工程化程度较低
- 没有验证集
- 没有指标评估
- 数据增强和路径处理还有可改进空间

如果后续要继续完善，最值得优先做的方向是：

1. 修正 `predict.py` 的跨平台路径问题
2. 修正 `dataset.py` 中数据增强的维度处理方式
3. 增加输入归一化
4. 引入验证集与 Dice / IoU 指标
5. 把训练参数改成命令行参数，提升可复用性

