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

### 3.3 网络定义
按照参考代码的注释要求，补全 Net 类的各层定义与前向传播过程：
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入通道1, 输出通道6, 卷积核5x5, padding=2
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # 输入通道6, 输出通道16, 卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层, 输入5*5*16, 输出120
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        # 全连接层, 输入120, 输出84
        self.fc2 = nn.Linear(120, 84)
        # 分类层, 输入84, 输出10
        self.clf = nn.Linear(84, 10)

    def forward(self, x):
        # conv1 -> sigmoid -> 平均池化 2x2, stride=2
        x = F.sigmoid(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # conv2 -> sigmoid -> 平均池化 2x2, stride=2
        x = F.sigmoid(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # 从第1维开始展平
        x = x.view(x.size(0), -1)
        # 全连接层1 -> sigmoid
        x = F.sigmoid(self.fc1(x))
        # 全连接层2 -> sigmoid
        x = F.sigmoid(self.fc2(x))
        # 分类层
        x = self.clf(x)
        return x
```
### 3.4 模型训练与测试
模型训练 30 个 epoch，每个 epoch 结束后在测试集上计算平均损失与分类准确率：
```
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

epochs = 30
accs, losses = [], []
for epoch in range(epochs):
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    correct = 0
    testloss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            testloss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    acc = correct / len(testloader.dataset)
    testloss = testloss / (batch_idx + 1)
    accs.append(acc)
    losses.append(testloss)
    print('epoch:{} loss:{:.4f} acc:{:.4f}'.format(epoch, testloss, acc))
```
### 3.5 特征图可视化
取测试集中的前 5 张图像，分别提取 conv1、conv2 后的特征图（conv2 前先对 conv1 输出做一次池化），并对通道维度求和后以灰度图显示：
```
feature1 = F.sigmoid(model.conv1(x))
feature2 = F.sigmoid(model.conv2(feature1))
feature2 = F.avg_pool2d(feature2, kernel_size=2, stride=2)

n = 5
img = x.detach().cpu().numpy()[:n]
feature_map1 = feature1.detach().cpu().numpy()[:n]
feature_map2 = feature2.detach().cpu().numpy()[:n]

fig, ax = plt.subplots(3, n, figsize=(10, 10))
for i in range(n):
    ax[0, i].imshow(img[i].sum(0), cmap='gray')
    ax[1, i].imshow(feature_map1[i].sum(0), cmap='gray')
    ax[2, i].imshow(feature_map2[i].sum(0), cmap='gray')
plt.show()
```
## 四、实验结果与分析
训练过程：随着 epoch 增加，测试集损失整体下降，准确率逐步提升并趋于稳定。Sigmoid 激活配合 Adam 优化器在 MNIST 上可取得较高的分类准确率。

特征图可视化分析：

### 1. 特征提取效果可视化展示
在完成网络训练并取得较高准确率后，我们选取测试集中的 5 个样本，提取其在第一层卷积（conv1）和第二层卷积（conv2）后的特征图，并将各通道特征图求和后进行可视化展示。
<img width="1681" height="353" alt="66752385c1035bf9c8849a5da9eead9" src="https://github.com/user-attachments/assets/25e9f3f6-bb5a-4d86-a5d7-72d53aa72ca8" />

<img width="1555" height="376" alt="55c985df39e008066bf7b62a258e678" src="https://github.com/user-attachments/assets/7e996691-6f7b-4420-b15e-bce214c0c31f" />

<img width="1590" height="343" alt="09cbbc990b67b36d259741879f264ca" src="https://github.com/user-attachments/assets/551cda47-49d6-43ed-be25-8bb12b8b653b" />

### 2.结果现象分析
通过观察上述特征图，可以得出以下现象：

第一行（原始输入）：展示了手写数字的原始灰度图，可以看出数字的边缘、笔画粗细和倾斜角度各有不同。

第二行（conv1 特征图）：在经过第一层卷积（1→6通道）并求和显示后，特征图依然保持了较高的空间分辨率（28×28）。其特征主要呈现为“粗线条”或“边缘增强”的效果。由于卷积核的局部感受野和 Sigmoid 激活函数的非线性映射，网络对数字笔画的边缘、轮廓等低级特征产生了显著的响应，数字的主体轮廓被高亮显示（白色），而背景区域则较暗（黑色）。

第三行（conv2 特征图）：经过两次卷积和池化操作后，特征图的空间尺寸缩小至 11×11（注：根据代码 feature2 未做第二次池化前的尺寸为 10×10，如显示 11×11 是 matplotlib 插值显示效果）。此时的特征图分辨率明显降低，像素颗粒感变强（模糊）。这说明网络在提取更高层特征时，已经抛弃了部分像素级细节，转而关注数字的整体形状、拓扑结构和语义信息。

### 3. 机理分析（结合网络结构）
从网络结构的角度解释上述现象：

感受野与层次特征：LeNet-5 中 conv1 的卷积核为 5×5，感受野较小，只能提取局部边缘特征；而 conv2 位于网络更深层，其感受野更大，能够组合低层边缘信息形成更抽象的形状特征。第三行特征图虽然模糊，但依然保留了数字的本质拓扑结构（如 8 的双环、5 的折角），证明了卷积网络“从低级特征到高级语义”的层次提取能力。

池化降维：平均池化层（avg_pool2d）在降低特征图尺寸的同时，保留了区域内的主要激活信息，这种“模糊化”处理实际上是一种特征降维，有助于减少计算量并提升模型对数字平移、缩放的鲁棒性。

Sigmoid 激活的作用：特征图呈现黑灰白渐变，正是 Sigmoid 将卷积输出的实数映射到 (0,1) 区间的结果，使得特征激活强烈的区域（白色）与未激活区域（黑色）形成对比，直观反映了不同卷积核对数字各个部位（如笔画的横折、圆弧）的敏感度。


## 五、实验小结
本实验基于 PyTorch 完整实现并训练了 LeNet-5 卷积神经网络，在 MNIST 手写数字数据集上完成了 30 个 epoch 的训练与测试，并通过可视化卷积层特征图直观观察了卷积层对不同层级特征的提取效果。实验加深了对卷积、池化、全连接层作用以及前向传播过程的理解，掌握了使用 PyTorch 搭建、训练 CNN 并进行特征可视化的基本方法，为后续学习更复杂的深度网络打下基础。





