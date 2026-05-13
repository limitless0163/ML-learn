import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# 下载训练集和测试集
data_path = './cifar10'
trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
testset = CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())
print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")
# trainset和testset可以直接用下标访问
# 每个样本为一个元组（data, label），其中data是一个3x32x32的张量，label是一个整数

# 可视化数据集
num_classes = 10
fig, axes = plt.subplots(num_classes, 10, figsize=(15, 15))
labels = np.array([t[1] for t in trainset])     # 取出所有样本的标签
for i in range(num_classes):
    indices = np.where(labels == i)[0]     # 找到标签为i的样本索引
    for j in range(10):
        # matplotlib绘制RGB图像时，图像矩阵依次是宽、高、颜色，与数据集中有差别
        # 因此需要用permute函数调整维度顺序
        axes[i][j].imshow(trainset[indices[j]][0].permute(1, 2, 0).numpy())     # 显示第j个样本
        # 去除坐标刻度
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])
plt.show()

class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        # 类别数目
        self.num_classes = num_classes
        # 输入通道数为3，输出通道数为32，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)     
        # 第二层卷积，输入通道和上一层的输出通道相同
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # 最大池化
        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        # 暂退，p表示每个位置被置为0的概率
        # 随机暂退只在训练时开启，在测试时应当关闭
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pooling2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=0.25)

        # 全连接层，输入特征数为64*8*8,与上一层的输出一致
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    # 前向传播，将输入按照顺序依次通过设置好的层
    def forward(self, x):
        x = F.relu(self.conv1(x))     
        x = F.relu(self.conv2(x))
        x = self.pooling1(x)     
        x = self.dropout1(x)     

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # 在全连接层之前，将x的形状转为(batch_size, n)
        x = x.view(len(x), -1)     
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)     
        return x

# 超参数设置
batch_size = 64
learning_rate = 1e-3
epochs = 5
np.random.seed(0)
torch.manual_seed(0)

# 批量生成器
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
model = CNN()
# 使用Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 使用交叉熵损失
criterion = F.cross_entropy

# 开始训练
for epoch in range(epochs):
    losses = 0
    accs = 0
    num = 0
    model.train()     # 设置模型为训练模式，启用暂退
    with tqdm(trainloader) as pbar:
        for data in pbar:
            images, labels = data
            outputs = model(images)     # 获取输出
            loss = criterion(outputs, labels)     # 计算损失
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累积损失
            num += len(labels)
            losses += loss.detach().numpy() * len(labels)
            # 准确率
            accs += (torch.argmax(outputs, dim=-1) == labels).sum().detach().numpy()
            pbar.set_postfix({
                'Epoch': epoch,
                'Train Loss': f'{losses / num:.3f}',
                'Train Acc': f'{accs / num:.3f}'
            })

# 计算模型在测试集上的准确率
losses = 0
accs = 0
num = 0
model.eval()     # 设置模型为评估模式，关闭暂退
with tqdm(testloader) as pbar:
    for data in pbar:
        images, labels = data
        outputs = model(images)
        loss = criterion(outputs, labels)
        num += len(labels)
        losses += loss.detach().numpy() * len(labels)
        accs += (torch.argmax(outputs, dim=-1) == labels).sum().detach().numpy()
        pbar.set_postfix({
            'Epoch': epoch,
            'Test Loss': f'{losses / num:.3f}',
            'Test Acc': f'{accs / num:.3f}'
        })


