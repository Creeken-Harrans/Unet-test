from unet_model import UNet
from dataset_with_val import split_train_val
from torch import optim
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt


def plot_loss_curves(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    ax.plot(
        epochs,
        train_losses,
        color="#2563eb",
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="Train Loss",
    )
    ax.plot(
        epochs,
        val_losses,
        color="#f97316",
        linewidth=2.4,
        marker="s",
        markersize=5,
        label="Validation Loss",
    )

    ax.scatter(
        best_epoch,
        best_val_loss,
        color="#dc2626",
        s=70,
        zorder=3,
        label="Best Val Loss",
    )
    ax.annotate(
        f"Best val_loss = {best_val_loss:.4f}\nEpoch = {best_epoch}",
        xy=(best_epoch, best_val_loss),
        xytext=(12, -28),
        textcoords="offset points",
        fontsize=10,
        color="#7f1d1d",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#fdba74", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.3),
    )

    ax.set_title("Training and Validation Loss", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def train_net(
    net, device, data_path, epochs=40, batch_size=1, lr=0.00001, train_ratio=0.7
):
    # 加载训练集和验证集
    train_dataset, val_dataset = split_train_val(data_path, train_ratio=train_ratio)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_val_loss统计，初始化为正无穷
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_loss = 0.0
        train_sample_count = 0
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            # 更新参数
            loss.backward()
            optimizer.step()
            batch_size_now = image.size(0)
            train_loss += loss.item() * batch_size_now
            train_sample_count += batch_size_now

        train_loss /= train_sample_count

        # 验证模式
        net.eval()
        val_loss = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                batch_size_now = image.size(0)
                val_loss += loss.item() * batch_size_now
                val_sample_count += batch_size_now

        val_loss /= val_sample_count
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}] train_loss: {train_loss:.6f} "
            f"val_loss: {val_loss:.6f}"
        )

        # 保存验证集loss最小的网络参数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), "best_model.pth")

    loss_curve_path = os.path.join(os.getcwd(), "result", "loss_curve_with_val.png")
    plot_loss_curves(train_losses, val_losses, loss_curve_path)
    print(f"损失曲线已保存: {loss_curve_path}")


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = (
        r"./Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/train"
    )
    train_net(net, device, data_path)
