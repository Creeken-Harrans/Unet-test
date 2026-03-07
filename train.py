from unet_model import UNet
from dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset=isbi_dataset, batch_size=batch_size, shuffle=True
    )
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float("inf")
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        epoch_loss = 0.0
        sample_count = 0
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
            epoch_loss += loss.item() * batch_size_now
            sample_count += batch_size_now

        epoch_loss /= sample_count
        print(f"Epoch [{epoch + 1}/{epochs}] train_loss: {epoch_loss:.6f}")

        # 保存loss值最小的网络参数
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(net.state_dict(), "best_model.pth")


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
