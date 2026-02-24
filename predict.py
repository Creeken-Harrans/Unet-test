import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load("best_model.pth", map_location=device))

    # 测试模式
    net.eval()

    # 1️⃣ 读取所有图片路径（要加上通配符 *.png）
    test_dir = (
        r"Deep-Learning-master\Deep-Learning-master\Pytorch-Seg\lesson-2\data\test"
    )
    tests_path = glob.glob(os.path.join(test_dir, "*.png"))
    print("测试图片数量:", len(tests_path))
    if len(tests_path) == 0:
        raise RuntimeError(f"在 {os.path.abspath(test_dir)} 下没有找到任何 .png 图片")

    # 2️⃣ 遍历所有图片
    for test_path in tests_path:
        print("正在处理:", test_path)

        # 保存结果地址
        # 当前脚本所在目录
        # 当前工作目录
        current_dir = os.getcwd()

        # result 文件夹路径
        result_dir = os.path.join(current_dir, "result")

        # 如果不存在就创建
        os.makedirs(result_dir, exist_ok=True)

        # 只取原文件名，不带路径
        file_name = os.path.basename(test_path)
        name_without_ext = os.path.splitext(file_name)[0]

        # 拼接当前目录下的新文件名
        save_res_path = os.path.join(result_dir, name_without_ext + "_res.png")

        # 读取图片
        img = cv2.imread(test_path)
        if img is None:
            raise RuntimeError(f"无法读取图片: {test_path}")

        # 3️⃣ 转为灰度图（OpenCV 默认是 BGR）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4️⃣ 转为 batch=1, channel=1 的形状 [1, 1, H, W]
        img = img.reshape(1, 1, img.shape[0], img.shape[1])

        # 5️⃣ 转为 tensor 并拷贝到 device
        img_tensor = torch.from_numpy(img).to(device=device, dtype=torch.float32)

        # 6️⃣ 预测（不需要梯度）
        with torch.no_grad():
            pred = net(img_tensor)
            # 如果网络最后一层没有 sigmoid（用了 BCEWithLogitsLoss），这一步很重要
            pred = torch.sigmoid(pred)

        # 7️⃣ 提取结果到 numpy，[N, C, H, W] -> [H, W]
        pred = pred.cpu().numpy()[0, 0]

        # 8️⃣ 阈值化到 0/255
        pred_bin = (pred >= 0.5).astype(np.uint8) * 255

        # 9️⃣ 保存图片
        cv2.imwrite(save_res_path, pred_bin)
        print("已保存结果:", save_res_path)
