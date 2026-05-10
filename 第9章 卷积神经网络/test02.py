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

from torchvision import models
import copy


# 定义图像处理方法
transform = transforms.Resize([512, 512])     # 将图像调整为512x512

def loadimg(path):
    # 加载路径为path的图像，形状为H*W*C
    img = plt.imread(path)     
    # 处理图像，注意重排维度使得通道维度在最前面
    img = transform(torch.tensor(img).permute(2, 0, 1))     
    # 展示图像
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()
    # 添加batch size维度
    img = img.unsqueeze(0).to(dtype=torch.float32)
    img = img / 255.0     # 将像素值归一化到[0, 1]范围
    return img

content_img_path = os.path.join('style_transfer', 'content', '04.jpg')
style_img_path = os.path.join('style_transfer', 'style.jpg')

# 加载内容图像
print('内容图像')
content_img = loadimg(content_img_path)
# 加载风格图像
print('风格图像')
style_img = loadimg(style_img_path)

# 内容损失
class ContentLoss(nn.Module):

    def __init__(self, target):
        # target为从目标图像中提取的内容特征
        super().__init__()
        # 我们不对target求梯度，因此将target从梯度的计算图中分离出来
        self.target = target.detach()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # 利用MSE计算输入图像与目标内容图像之间的损失
        self.loss = self.criterion(x.clone(), self.target)
        return x        # 只计算损失，不改变输入
    
    def backward(self):
        # 由于本模块只包含损失计算，不改变输入，因此要单独定义反向传播
        self.loss.backward(retain_graph=True)
        return self.loss
    
def gram(x):
    # 计算G矩阵
    batch_size, n, w, h = x.shape
    f = x.view(batch_size * n, w * h)     # 将特征图展平为二维矩阵
    g = f @ f.T / (batch_size * n * w * h)     # 计算G矩阵，并进行归一化
    return g

# 风格损失
class StyleLoss(nn.Module):

    def __init__(self, target):
        # target为从目标图像中提取的风格特征
        # weight为设置的强度系数lambda
        super().__init__()
        self.target_gram = gram(target.detach())     # 目标的格拉姆矩阵
        self.criterion = nn.MSELoss()

    def forward(self, x):
        input_gram = gram(x.clone())     # 输入图像的格拉姆矩阵
        self.loss = self.criterion(input_gram, self.target_gram)     # 计算损失
        return x        # 只计算损失，不改变输入
    
    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss
    
vgg16 = models.vgg16(weights=True).features     # 加载预训练的VGG16模型
# 选定用于提取特征的卷积层，Conv_13对应着第五块的第三个卷积层
content_layer = ['Conv_13']
# 下面这些层分别对应第一至第五块的第一个卷积层
style_layer = ['Conv_1', 'Conv_3', 'Conv_5', 'Conv_8', 'Conv_11']

content_losses = []
style_losses = []

model = nn.Sequential()    # 存储新模型的层
vgg16 = copy.deepcopy(vgg16)     
index = 1    # 计数卷积层

# 遍历VGG16的网络结构，选取需要的层
for layer in list(vgg16):
    if isinstance(layer, nn.Conv2d):
        name = 'Conv_' + str(index)
        model.append(layer)
        if name in content_layer:
            # 如果当前层是内容层，则添加内容损失模块
            target = model(content_img).clone()     # 从内容图像中提取特征
            content_loss = ContentLoss(target)      # 创建内容损失模块
            model.append(content_loss)              # 将内容损失模块添加到模型中
            content_losses.append(content_loss)     # 将内容损失模块保存到列表中

        if name in style_layer:
            # 如果当前层是风格层，则添加风格损失模块
            target = model(style_img).clone()       # 从风格图像中提取特征
            style_loss = StyleLoss(target)          # 创建风格损失模块
            model.append(style_loss)                # 将风格损失模块添加到模型中
            style_losses.append(style_loss)         # 将风格损失模块保存到列表中

if isinstance(layer, nn.ReLU):      # 如果是激活函数层
    model.append(layer)
    index += 1

if isinstance(layer, nn.MaxPool2d):     # 如果是池化层
    model.append(layer)

# 超参数
epochs = 200
learning_rate = 0.1
lbd = 1e6       # 强度系数

input_img = content_img.clone()   # 从内容图像开始迁移
param = nn.Parameter(input_img.data)     # 将输入图像作为参数进行优化
optimizer = torch.optim.Adam([param], lr=learning_rate)     # 使用Adam优化器

for i in range(epochs):
    style_score = 0     # 本轮的风格损失
    content_score = 0   # 本轮的内容损失
    model(param)       # 前向传播，计算损失
    for cl in content_losses:
        content_score += cl.backward()     # 反向传播，计算内容损失
    for sl in style_losses:
        style_score += sl.backward()       # 反向传播，计算风格损失
style_score *= lbd     # 调整风格损失的强度
loss = content_score + style_score     # 总损失
# 更新输入图像
optimizer.zero_grad()
loss.backward()
optimizer.step()
# 每次对输入图像进行更新后，图像中部分像素点可能会超出[0, 1]范围，因此需要进行裁剪
param.data.clamp_(0, 1)

if i % 50 == 0 or i == epochs - 1:
    print(f'训练轮数：{i},\t内容损失：{content_score.item():.4f},\t风格损失：{style_score.item():.4f}')
    plt.imshow(input_img[0].permute(1, 2, 0).numpy())
    plt.show()