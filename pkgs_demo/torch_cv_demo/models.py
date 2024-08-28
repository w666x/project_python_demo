# -*- coding: utf-8 -*-
"""模型设置"""

# +
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数3，输出通道数6，卷积核大小5
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，核大小2，步长2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数6，输出通道数16，卷积核大小5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层，输入维度16*5*5，输出维度120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入维度120，输出维度84
        self.fc3 = nn.Linear(84, 10)  # 全连接层，输入维度84，输出维度10（CIFAR10有10类）

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积+ReLU激活函数+池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积+ReLU激活函数+池化
        x = x.view(-1, 16 * 5 * 5)  # 将特征图展平
        x = F.relu(self.fc1(x))  # 第一层全连接+ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二层全连接+ReLU激活函数
        x = self.fc3(x)  # 第三层全连接
        return x

if __name__ == "__main__":
    net = Net()
    print(net)
