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

### 第一行输入图像展示了原始手写数字；

### 第二行为 conv1 后的特征图，可观察到网络对数字的笔划边缘、轮廓产生了响应，部分通道突出笔画区域，体现了卷积核对边缘等低级特征的提取能力；

### 第三行为经过 conv1 池化与 conv2 后的特征图，尺寸由 28×28 降至 5×5，分辨率降低但语义信息更抽象，反映网络逐层从低级边缘特征向高级语义特征过渡的过程。

网络结构合理性：conv1 使用 padding=2 使特征图尺寸保持不变（28×28），便于保留边缘信息；两次 2×2 平均池化将空间尺寸依次减半为 14×14 和 5×5，与全连接层输入 5×5×16=400 相匹配，验证了网络各层参数设计的正确性。

## 五、实验小结
本实验基于 PyTorch 完整实现并训练了 LeNet-5 卷积神经网络，在 MNIST 手写数字数据集上完成了 30 个 epoch 的训练与测试，并通过可视化卷积层特征图直观观察了卷积层对不同层级特征的提取效果。实验加深了对卷积、池化、全连接层作用以及前向传播过程的理解，掌握了使用 PyTorch 搭建、训练 CNN 并进行特征可视化的基本方法，为后续学习更复杂的深度网络打下基础。





