import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path, imgs_path=None, use_augment=True):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.use_augment = use_augment
        if imgs_path is None:
            # 拼接完整的搜索路径
            search_pattern = os.path.join(data_path, "image", "*.png")
            print("正在搜索的路径:", search_pattern)
            # 执行搜索
            self.imgs_path = sorted(
                glob.glob(search_pattern)
            )  # 返回一个列表，包含所有匹配的文件路径
        else:
            self.imgs_path = sorted(imgs_path)
        print("找到的图片列表:", self.imgs_path)
        print("找到的图片数量:", len(self.imgs_path))
        print("是否启用数据增强:", self.use_augment)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace("image", "label")
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # 增加图片读取失败的检查
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        if label is None:
            raise ValueError(f"无法读取标签: {label_path}")

        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 这里出来的像素值是0-255的整数
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = (
                label / 255
            )  # 为什么只处理标签, 因为标签的像素值是0和255，代表两类，而模型输出的是0和1，所以需要将标签的像素值也转换为0和1。训练图片的像素值是0-255的整数，代表灰度值，不需要转换。
        # 训练集随机进行数据增强，验证集不做增强
        if self.use_augment:
            flipCode = random.choice([-1, 0, 1, 2])
            if flipCode != 2:
                image = self.augment(image, flipCode)
                label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def split_train_val(data_path, train_ratio=0.7, seed=42):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")

    search_pattern = os.path.join(data_path, "image", "*.png")
    imgs_path = sorted(glob.glob(search_pattern))
    print("用于划分训练/验证集的图片数量:", len(imgs_path))

    if len(imgs_path) < 2:
        raise ValueError("图片数量不足，至少需要2张图片才能划分训练集和验证集")

    shuffled_imgs_path = imgs_path[:]
    random.Random(seed).shuffle(shuffled_imgs_path)

    train_size = int(len(shuffled_imgs_path) * train_ratio)
    train_size = max(1, min(train_size, len(shuffled_imgs_path) - 1))

    train_imgs_path = shuffled_imgs_path[:train_size]
    val_imgs_path = shuffled_imgs_path[train_size:]

    print("训练集数量:", len(train_imgs_path))
    print("验证集数量:", len(val_imgs_path))

    train_dataset = ISBI_Loader(
        data_path, imgs_path=train_imgs_path, use_augment=True
    )
    val_dataset = ISBI_Loader(data_path, imgs_path=val_imgs_path, use_augment=False)
    return train_dataset, val_dataset


if __name__ == "__main__":
    # 这里改成你的绝对路径
    train_dataset, val_dataset = split_train_val(
        r"./Deep-Learning-master/Deep-Learning-master/Pytorch-Seg/lesson-2/data/train"
    )
    print("训练集个数：", len(train_dataset))
    print("验证集个数：", len(val_dataset))

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("错误：没有找到任何图片，请检查路径和目录结构！")
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=2, shuffle=True
        )
        for image, label in train_loader:
            print(image.shape)
