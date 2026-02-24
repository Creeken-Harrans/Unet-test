import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # 拼接完整的搜索路径
        search_pattern = os.path.join(data_path, "image/*.png")
        print("正在搜索的路径:", search_pattern)
        # 执行搜索
        self.imgs_path = glob.glob(
            search_pattern
        )  # 返回一个列表，包含所有匹配的文件路径
        print("找到的图片列表:", self.imgs_path)
        print("找到的图片数量:", len(self.imgs_path))

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
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    # 这里改成你的绝对路径
    isbi_dataset = ISBI_Loader(
        r"Deep-Learning-master\Deep-Learning-master\Pytorch-Seg\lesson-2\data\train"
    )
    print("数据个数：", len(isbi_dataset))

    if len(isbi_dataset) == 0:
        print("错误：没有找到任何图片，请检查路径和目录结构！")
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=isbi_dataset, batch_size=2, shuffle=True
        )
        for image, label in train_loader:
            print(image.shape)
