# The edge code refers to 'Non-Local Deep Features for Salient Object Detection', CVPR 2017.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5


def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad


def scharr_edges(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径！")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img)

    # 对每个通道应用 Scharr 算子
    scharr_R = cv2.Scharr(R, cv2.CV_64F, 1, 0) + cv2.Scharr(R, cv2.CV_64F, 0, 1)
    scharr_G = cv2.Scharr(G, cv2.CV_64F, 1, 0) + cv2.Scharr(G, cv2.CV_64F, 0, 1)
    scharr_B = cv2.Scharr(B, cv2.CV_64F, 1, 0) + cv2.Scharr(B, cv2.CV_64F, 0, 1)

    # 转换为 uint8 格式
    scharr_R = cv2.convertScaleAbs(scharr_R)
    scharr_G = cv2.convertScaleAbs(scharr_G)
    scharr_B = cv2.convertScaleAbs(scharr_B)

    # 合并结果
    combined_scharr = cv2.merge((scharr_R, scharr_G, scharr_B))

    # 显示结果
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Scharr Edges")
    plt.imshow(combined_scharr)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def edge_detection_batch(tensor, method='laplacian', low_threshold=100, high_threshold=200):
    """
    对 batch_size * 3 * H * W 的张量进行边缘检测。

    参数：
        tensor (torch.Tensor): 输入张量，形状为 batch_size * 3 * H * W。
        method (str): 边缘检测方法，支持 'canny', 'sobel', 'laplacian'。
        low_threshold (int): Canny 算法低阈值，仅在 method='canny' 时有效。
        high_threshold (int): Canny 算法高阈值，仅在 method='canny' 时有效。

    返回：
        torch.Tensor: 边缘提取后的张量，形状与输入一致。
    """
    # 检查输入维度
    if tensor.dim() != 4 or tensor.size(1) != 3:
        raise ValueError("输入张量应为 batch_size * 3 * H * W 格式。")

    batch_size, channels, height, width = tensor.shape
    edges_batch = []

    # 对 batch 中的每张图像分别处理
    for i in range(batch_size):
        edges_channels = []

        # 对每个通道进行边缘检测
        for c in range(channels):
            # 提取单通道图像 (H, W)
            img = tensor[i, c].cpu().numpy()  # 转为 NumPy 数组
            img = (img * 255).astype(np.uint8)  # 假设输入范围是 [0, 1]

            if method == 'canny':
                edges = cv2.Canny(img, low_threshold, high_threshold)
            elif method == 'sobel':
                edges = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
                edges = cv2.convertScaleAbs(edges)  # 转为 uint8
            elif method == 'laplacian':
                edges = cv2.Laplacian(img, cv2.CV_64F)
                edges = cv2.convertScaleAbs(edges)  # 转为 uint8
            else:
                raise ValueError(f"不支持的边缘检测方法：{method}")

            edges_channels.append(edges / 255.0)  # 归一化回 [0, 1]

        # 合并通道
        edges_image = np.stack(edges_channels, axis=0)  # (3, H, W)
        edges_batch.append(edges_image)

    # 合并 batch
    edges_batch = np.stack(edges_batch, axis=0)  # (batch_size, 3, H, W)

    return torch.tensor(edges_batch, dtype=torch.float32, device=tensor.device)


def pred_edge_prediction(pred):
    # infer edge from prediction
    pred = F.pad(pred, (1, 1, 1, 1), mode='replicate')
    pred_fx = F.conv2d(pred, fx)
    pred_fy = F.conv2d(pred, fy)
    pred_grad = (pred_fx*pred_fx + pred_fy*pred_fy).sqrt().tanh()

    return pred_fx, pred_fy, pred_grad


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return np.mean(self.losses[np.maximum(len(self.losses)-self.num, 0):])


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))

