# -*- coding: utf-8 -*-
"""数据loader构造"""

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def my_data_loader():
    """定义数据预处理操作"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 数据增强：随机翻转图片
        transforms.RandomCrop(32, padding=4),  # 数据增强：随机裁剪图片
        transforms.ToTensor(),  # 将PIL.Image或者numpy.ndarray数据类型转化为torch.FloadTensor，并归一化到[0.0, 1.0]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化（这里的均值和标准差是CIFAR10数据集的）
    ])

    # 下载并加载训练数据集
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 下载并加载测试数据集
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, trainloader
    

if __name__ == "__main__":
    trainloader, testloader = my_data_loader()
    trainloader.dataset.data.shape, trainloader.dataset.targets.__len__()
