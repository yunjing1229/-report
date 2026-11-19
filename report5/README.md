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




