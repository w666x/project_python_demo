# -*- coding: utf-8 -*-
"""目标检测-CIFAR10数据集来训练一个卷积神经网络"""

# +
from matplotlib.pyplot import imshow
import torch
from torch import nn, optim
import torchvision

from data_loader import my_data_loader
from models import Net


# -

def test_eval(testloader):
    """模型效果测试"""
    dataiter = iter(testloader)
    images, labels = dataiter.next()
#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == "__main__":
    trainloader, testloader = my_data_loader()
    net = Net()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # 在数据集上训练两遍
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个批次打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    test_eval(testloader)
    # 保存模型
    torch.save(net.state_dict(), './cifar_net.pth')

    # 加载模型
    net = Net()  # 创建新的网络实例
    net.load_state_dict(torch.load('./cifar_net.pth'))  # 加载模型参数
    test_eval(testloader)
