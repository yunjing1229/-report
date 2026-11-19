# 实验五：卷积神经网络 LeNet 实现与特征图可视化
## 一、实验目的
掌握 PyTorch 中卷积神经网络的基本搭建方法；

理解 LeNet-5 的网络结构，包括卷积层、池化层、全连接层的作用；

学会在 MNIST 手写数字数据集上完成模型的训练与测试；

学会提取并可视化卷积层的特征图，直观理解卷积层对图像特征的提取过程。
## 二、实验环境
编程语言：Python

深度学习框架：PyTorch

数据集与工具：torchvision

可视化工具：Matplotlib

计算设备：自动选择 CUDA 或 CPU
## 三、实验内容与原理
### 3.1 LeNet-5 网络结构
LeNet-5 是经典的卷积神经网络，主要用于手写数字识别。本实验实现的网络结构如下：
<img width="1599" height="593" alt="image" src="https://github.com/user-attachments/assets/3770ece1-7c43-482a-ba2e-a69e428c0a06" />
激活函数全部采用 Sigmoid，损失函数采用交叉熵损失（Cross Entropy），优化器采用 Adam，学习率设为 1e-2。
### 3.2 数据准备
使用 torchvision 加载 MNIST 数据集，训练集与测试集均通过 transforms.ToTensor() 转换为张量，批大小设为 512，训练集打乱顺序。
```
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
```




