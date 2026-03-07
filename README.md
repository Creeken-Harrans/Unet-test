# U-Net 图像分割实验说明

## 1. 项目概述

这个工作目录是一个基于 U-Net 的二值图像分割实验区，核心目标是完成：

1. 使用单通道输入图像训练一个二分类分割模型。
2. 对测试图片进行预测，输出二值分割结果。
3. 对比两种不同的训练优化策略：
   - 方式一：只依据 `train_loss` 训练和选模。
   - 方式二：引入验证集，使用 `train_loss` 训练、使用 `val_loss` 监控泛化并选模。

当前根目录下的脚本是这个实验真正使用的入口文件。`Deep-Learning-master/` 子目录保留了上游参考仓库和数据，其中本项目实际使用的数据位于：

`Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/`

按当前仓库内容统计，样例数据包含：

- 训练图像：30 张
- 训练标签：30 张
- 测试图像：30 张

## 2. 工作目录说明

根目录核心文件如下：

- `train.py`
  - 基础训练脚本。
  - 使用全部训练数据。
  - 只输出每个 epoch 的 `train_loss`。
  - 以训练集平均损失最小作为模型保存标准。

- `dataset.py`
  - `train.py` 对应的数据加载脚本。
  - 负责读取训练图像、标签、灰度转换、标签归一化和随机翻转增强。

- `train_with_val.py`
  - 带验证集的训练脚本。
  - 将训练数据拆分为训练集和验证集。
  - 每个 epoch 输出 `train_loss` 和 `val_loss`。
  - 训练结束后绘制两条 loss 曲线到同一张图中。

- `dataset_with_val.py`
  - `train_with_val.py` 对应的数据加载脚本。
  - 支持按 `0.7 / 0.3` 划分训练集和验证集。
  - 训练集启用随机增强，验证集关闭增强。

- `predict.py`
  - 推理脚本。
  - 加载根目录下的 `best_model.pth`。
  - 对测试集图片逐张预测，并将结果保存到 `result/`。

- `unet_model.py`
  - U-Net 主体网络定义。

- `unet_parts.py`
  - U-Net 组成模块定义，包括 `DoubleConv`、`Down`、`Up`、`OutConv`。

- `best_model.pth`
  - 当前保存在根目录下的模型权重文件。
  - 会被 `train.py` 或 `train_with_val.py` 覆盖。

- `result/`
  - 推理结果和损失曲线输出目录。

## 3. 数据组织方式

本项目使用的训练数据结构如下：

```text
Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/
├── train/
│   ├── image/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── label/
│       ├── 0.png
│       ├── 1.png
│       └── ...
└── test/
    ├── 0.png
    ├── 1.png
    └── ...
```

约定如下：

- `train/image/` 与 `train/label/` 中的文件名一一对应。
- 例如 `train/image/7.png` 对应标签 `train/label/7.png`。
- 标签图是二值图，像素通常为 `0` 或 `255`。
- 脚本内部会将标签从 `0/255` 转换为 `0/1`。
- 输入图像会被转为灰度图，并 reshape 为 `[1, H, W]`。

## 4. 模型结构

模型定义在 `unet_model.py` 和 `unet_parts.py` 中，使用的是标准 U-Net 结构：

- 编码器部分：
  - `DoubleConv`
  - `Down`
- 解码器部分：
  - `Up`
  - `OutConv`

当前网络初始化方式为：

```python
net = UNet(n_channels=1, n_classes=1)
```

说明：

- `n_channels=1`：输入为单通道灰度图。
- `n_classes=1`：输出单通道 mask，对应二分类分割。

训练时使用：

- 优化器：`RMSprop`
- 损失函数：`BCEWithLogitsLoss`

这意味着模型输出的是 logits，不是已经过 sigmoid 的概率图。因此：

- 训练时直接把 logits 送入 `BCEWithLogitsLoss`
- 预测时需要手动执行 `torch.sigmoid()`

## 5. 数据预处理与增强

### 5.1 `dataset.py`

基础训练方式的数据处理逻辑：

1. 读取 `train/image/*.png`
2. 根据路径把 `image` 替换为 `label` 找到标签
3. 使用 OpenCV 读取图像
4. 将图像和标签都转换为灰度图
5. 将标签从 `255` 转成 `1`
6. 随机执行翻转增强

随机翻转的候选为：

- `1`：水平翻转
- `0`：垂直翻转
- `-1`：水平 + 垂直翻转
- `2`：不做增强

### 5.2 `dataset_with_val.py`

验证集版本在基础数据处理上做了两件额外的事：

1. 支持传入已经划分好的文件列表。
2. 增加 `use_augment` 开关，保证：
   - 训练集启用随机增强
   - 验证集关闭增强

这是合理的，因为验证集的目标是稳定评估模型泛化能力，不应再对验证样本做随机扰动。

## 6. 两种训练优化方式

这部分是整个项目最重要的区别。

需要先强调一点：

这两种方式虽然“优化策略”不同，但它们的底层优化器仍然都是 `RMSprop`，损失函数仍然都是 `BCEWithLogitsLoss`。真正不同的是：

- 参数更新依据什么损失
- 模型保存依据什么指标
- 是否引入验证集监控过拟合

### 6.1 方式一：仅基于训练集损失的优化

对应文件：

- `train.py`
- `dataset.py`

#### 工作流程

1. 读取全部训练数据。
2. 不划分验证集。
3. 对每个 batch 计算训练损失。
4. 使用训练损失反向传播并更新参数。
5. 每个 epoch 结束后计算该轮平均 `train_loss`。
6. 输出：

```text
Epoch [19/40] train_loss: 0.204504
```

7. 如果当前 epoch 的平均 `train_loss` 更小，就保存为新的 `best_model.pth`。

#### 关键特点

- 反向传播的载体：`train_loss`
- 模型选择标准：最小 `train_loss`
- 使用数据量：100% 训练数据都参与参数更新

#### 优点

- 流程简单，容易理解。
- 全部训练样本都用于更新参数。
- 适合快速验证代码能否正常训练、loss 是否下降。

#### 风险

- 没有验证集，无法判断泛化能力。
- 即使训练损失持续下降，也可能已经过拟合。
- 按训练损失选出的“最佳模型”不一定是实际泛化最好的模型。

#### 适用场景

- 快速调通代码
- 初步验证模型是否可收敛
- 数据量非常小，暂时不想切分验证集

### 6.2 方式二：基于训练集更新 + 验证集监控的优化

对应文件：

- `train_with_val.py`
- `dataset_with_val.py`

#### 工作流程

1. 从 `train/image/*.png` 中读取全部训练样本。
2. 使用固定随机种子 `seed=42` 做可复现划分。
3. 按 `0.7 / 0.3` 拆分训练集和验证集。
4. 当前仓库有 30 张训练图，因此当前实际划分为：
   - 训练集：21 张
   - 验证集：9 张
5. 训练集启用随机增强，验证集不增强。
6. 每个 epoch 中：
   - 用训练集计算 `train_loss`
   - 用 `train_loss.backward()` 做反向传播
   - 再用验证集计算 `val_loss`
7. 输出：

```text
Epoch [19/40] train_loss: 0.204504 val_loss: 0.217310
```

8. 使用最小 `val_loss` 作为模型保存标准。
9. 训练结束后绘制 `train_loss` 与 `val_loss` 的同图折线图，并保存到：

```text
result/loss_curve_with_val.png
```

#### 一个非常重要的概念

在这种方式里：

- `train_loss` 参与反向传播
- `val_loss` 不参与反向传播

也就是说，验证损失不会直接更新模型参数。它的作用是：

- 监控泛化能力
- 判断是否出现过拟合
- 选择验证集表现最好的模型权重

因此更准确的说法是：

这种方式不是“让 `val_loss` 参与梯度下降”，而是“让 `val_loss` 参与模型评估与模型选择”。

#### 优点

- 可以观察训练集和验证集之间的差异。
- 更容易发现过拟合。
- 按 `val_loss` 选模，通常比按 `train_loss` 选模更可靠。
- 能输出 loss 曲线，更直观。

#### 风险与代价

- 训练样本被切分后，真正参与更新参数的数据会变少。
- 当前数据只有 30 张，验证集波动可能会比较大。
- 结果对划分方式有一定敏感性。

#### 适用场景

- 正式比较模型效果
- 希望监控过拟合
- 希望保留更有泛化能力的模型

## 7. 两种方式的对比总结

| 对比项 | 方式一：`train.py` | 方式二：`train_with_val.py` |
| --- | --- | --- |
| 数据划分 | 不划分验证集 | 按 0.7 / 0.3 划分训练/验证 |
| 反向传播依据 | `train_loss` | `train_loss` |
| 是否计算 `val_loss` | 否 | 是 |
| 模型保存标准 | 最小 `train_loss` | 最小 `val_loss` |
| 是否更利于发现过拟合 | 否 | 是 |
| 是否绘制 loss 曲线 | 否 | 是 |
| 训练数据利用率 | 100% 用于更新参数 | 70% 用于更新参数 |
| 适合用途 | 快速调试、初步收敛验证 | 更正式的训练与选模 |

## 8. 运行环境

推荐环境：

- Python 3.9 及以上
- PyTorch
- NumPy
- OpenCV
- Matplotlib

可参考安装：

```bash
pip install torch numpy opencv-python matplotlib
```

如果需要 GPU 版 PyTorch，请按你的 CUDA 版本安装对应官方包。

## 9. 运行方式

请在项目根目录运行以下命令。

### 9.1 基础训练

```bash
python train.py
```

执行效果：

- 每轮输出一次平均 `train_loss`
- 将训练损失最优的模型保存为 `best_model.pth`

### 9.2 带验证集训练

```bash
python train_with_val.py
```

执行效果：

- 每轮输出一次 `train_loss` 和 `val_loss`
- 将验证损失最优的模型保存为 `best_model.pth`
- 生成 `result/loss_curve_with_val.png`

### 9.3 预测

```bash
python predict.py
```

执行效果：

1. 从根目录读取 `best_model.pth`
2. 读取测试目录：

```text
Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/test/
```

3. 对每张图进行预测
4. 保存到：

```text
result/*_res.png
```

## 10. 输出文件说明

### 10.1 `best_model.pth`

这是当前项目的模型权重文件。它本质上是一个权重快照，也可以理解为当前项目中最常用的 checkpoint。

注意：

- `train.py` 和 `train_with_val.py` 都会写这个文件。
- 如果你先运行一种训练方式，再运行另一种训练方式，前一个模型会被覆盖。
- 如果你要对比两种训练策略的结果，建议手动改名保存，例如：
  - `best_model_train_only.pth`
  - `best_model_with_val.pth`

### 10.2 `result/loss_curve_with_val.png`

这是验证集训练方式生成的损失曲线图：

- 蓝线：`train_loss`
- 橙线：`val_loss`
- 红点：最优 `val_loss` 对应的 epoch

这张图最适合用来观察：

- 是否正常收敛
- 是否出现训练集继续下降但验证集开始反弹的过拟合迹象

### 10.3 `result/*_res.png`

这是推理后的二值分割结果图。输出逻辑是：

1. 模型输出 logits
2. 经过 `sigmoid`
3. 以 `0.5` 为阈值
4. 转成 `0/255` 的二值图后保存

## 11. 代码逻辑上的关键理解

### 11.1 为什么训练时不用 `sigmoid`

因为训练使用的是 `BCEWithLogitsLoss`。这个损失函数已经把 sigmoid 与二值交叉熵合并在一起，数值上更稳定。

### 11.2 为什么预测时要 `sigmoid`

因为推理阶段需要把 logits 转换为概率，再进行阈值化。

### 11.3 验证集损失是否参与反向传播

不参与。

项目当前的正确理解应该是：

- 参数更新由 `train_loss` 驱动
- 验证集只负责评估和选模

这也是标准做法。

### 11.4 为什么方式二更有助于防止过拟合

严格来说，方式二不是直接“阻止”过拟合，而是：

1. 让你看到 `val_loss` 是否变差
2. 让你保留验证集表现最好的模型

因此它是通过“监控和选模”来降低过拟合风险，而不是让验证损失参与梯度更新。

## 12. 常见注意事项

### 12.1 请尽量从项目根目录运行脚本

原因：

- `train.py` 和 `train_with_val.py` 中的数据路径是相对路径
- `predict.py` 会把结果写到当前工作目录下的 `result/`

如果不从根目录运行，输出位置和相对路径可能不符合预期。

### 12.2 数据路径写死在脚本中

当前脚本默认使用：

```text
./Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/train
```

如果你要换自己的数据集，需要同步修改：

- `train.py`
- `train_with_val.py`
- `predict.py` 中的测试目录

如果你的新数据集目录结构不再是下面这种形式：

```text
train/
├── image/*.png
└── label/*.png
```

那么还需要再修改：

- `dataset.py`
- `dataset_with_val.py`

### 12.3 当前输入图像没有额外归一化

当前实现会把图像转成灰度图，但不会进一步把输入缩放到 `0~1`。这对跑通流程没有问题，但如果后续你想进一步提升训练稳定性，可以考虑增加输入归一化。

### 12.4 当前还没有 early stopping

`train_with_val.py` 虽然已经能观察 `val_loss`，但还没有实现：

- early stopping
- 学习率调度器
- 更复杂的数据增强

这些都可以作为下一步优化项。

## 13. 建议的使用顺序

如果你第一次接手这个目录，推荐按下面顺序使用：

1. 先运行 `python train.py`
   - 确认环境、数据路径、模型训练流程都正常
2. 再运行 `python train_with_val.py`
   - 观察 `train_loss` 和 `val_loss`
   - 查看 `result/loss_curve_with_val.png`
3. 最后运行 `python predict.py`
   - 检查预测结果是否符合预期

## 14. 总结

这个工作目录本质上提供了同一个 U-Net 分割任务的两套训练方案：

- `train.py`
  - 更简单
  - 更适合快速调试
  - 只围绕 `train_loss` 优化和选模

- `train_with_val.py`
  - 更完整
  - 更适合正式训练
  - 用 `train_loss` 训练，用 `val_loss` 监控泛化并选择更可靠的模型

如果你的目标是“先跑通”，优先使用 `train.py`。

如果你的目标是“更合理地评估模型并减少过拟合风险”，优先使用 `train_with_val.py`。
