# 实验四：基于多层感知机的MNIST手写数字识别
## 一、实验目的
学会深度学习框架PyTorch的基本使用方法，利用PyTorch构建多层感知机（MLP）神经网络，对MNIST手写数字数据集进行训练与测试，实现图像分类，并理解前向传播、损失计算、反向传播及模型评估的基本流程。
## 二、实验内容
### 2.1 导入深度学习相关的依赖包
1. PyTorch库：深度学习基础框架，用于构建网络、自动求导和GPU加速。

2. torchvision库：计算机视觉数据集与图像变换工具，用于下载和预处理MNIST数据集。

3. NumPy库：科学计算的基础库。

4. Matplotlib库：绘图与数据可视化。

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```
<img width="675" height="276" alt="image" src="https://github.com/user-attachments/assets/8d4c9620-d464-42f2-b384-8203b42c7cca" />

### 2.2 数据集加载与预处理
设置批处理大小（batch_size = 2048），并检测当前设备是否支持CUDA（GPU加速）。通过torchvision.datasets.MNIST下载训练集与测试集，并使用transforms.ToTensor()将图像转换为张量格式。最后使用DataLoader构建数据加载器，设置shuffle=True打乱数据顺序。
```
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
```
### 2.3 构建多层感知机（MLP）模型
定义一个继承自nn.Module的MLP类。输入层为784维（对应28×28的MNIST图像展平），隐藏层为128维，输出层为10维（对应0-9十个类别）。隐藏层使用ReLU激活函数。
```
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
        
    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2

model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
### 2.4 模型训练与测试
设置训练轮数（epochs = 30）。在每个epoch中，先调用model.train()进入训练模式，遍历训练集，将图像展平为[batch_size, 784]并送入模型，计算交叉熵损失，执行反向传播并更新参数。随后调用model.eval()进入评估模式，在测试集上计算平均损失和准确率。
```
epochs = 30
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            
    test_loss = test_loss / (batch_idx + 1)
    acc = correct / len(test_loader.dataset)
    print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, test_loss, acc))
```
## 实验结果与分析
<img width="519" height="785" alt="image" src="https://github.com/user-attachments/assets/5a995e42-a08a-4a77-8c34-446705c61285" />

### 1. 模型结构与参数
输入层：784个神经元（28×28像素展平）

隐藏层：128个神经元，使用ReLU激活函数

输出层：10个神经元（对应0-9分类）

优化器：随机梯度下降（SGD），学习率设为0.01

损失函数：交叉熵损失（Cross Entropy Loss）
### 2. 训练过程分析
通过控制台输出的日志可以看到，随着epoch的增加，测试集上的损失（loss）逐渐下降，准确率（acc）逐渐上升。由于MLP结构较简单且学习率适中，模型在训练初期收敛较快，后期趋于稳定。经过30个epoch的训练，模型在MNIST测试集上的准确率通常可以达到97%以上
### 3. 模型评估与总结
准确率：模型在测试集上表现良好，说明MLP能够有效提取图像中的特征并进行分类。

训练效率：使用batch_size=2048可以充分利用GPU的并行计算能力，加快训练速度；使用torch.no_grad()在测试阶段关闭梯度计算，节省了内存和计算资源。

局限性分析：虽然MLP在MNIST上表现优异，但它将图像展平为一维向量，丢失了图像的空间结构信息（如局部相关性）。对于更复杂的图像任务，通常需要采用卷积神经网络（CNN）来提取更丰富的空间特征。
## 四、实验小结
本实验完成了MNIST手写数字数据集的加载、预处理，以及多层感知机（MLP）模型的构建、训练与测试。

实验结果表明：
```
使用PyTorch可以快速构建和训练神经网络，自动求导机制大大简化了反向传播的实现。

交叉熵损失函数配合SGD优化器，能够有效指导模型参数更新，使模型在测试集上达到较高的分类准确率。

通过model.train()和model.eval()的切换，正确管理了训练和评估阶段的行为（如Dropout、BatchNorm等，虽然本实验未使用，但这是良好的编程习惯）。
```
通过本次实验，进一步加深了对深度学习基本流程（前向传播、损失计算、反向传播、参数更新）的理解，掌握了PyTorch的基本使用方法，为后续学习卷积神经网络（CNN）及更复杂的图像分类任务奠定了基础。
