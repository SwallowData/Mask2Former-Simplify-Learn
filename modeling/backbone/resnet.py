#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   resnet.py
@Time    :   2022/04/23 14:08:10
@Author  :   BQH
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   Backbone - ResNet实现文件，用于图像分割任务的主干网络
'''

# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from addict import Dict  # 字典增强库，提供更方便的字典操作
import torch.utils.model_zoo as model_zoo  # 用于从网络加载预训练模型

# 定义Batch Normalization的动量参数，用于控制统计参数的更新速度
BN_MOMENTUM = 0.1

# 预训练模型的下载地址字典
model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3卷积核的卷积操作，带padding，用于保持特征图尺寸"""
    # 创建3x3卷积层，padding=1保证输出尺寸与输入尺寸在相同stride下保持一致关系
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class InvertedResidual(nn.Module):
    """倒残差结构块，用于轻量化网络设计，常用于MobileNet等网络中"""

    def __init__(self, in_channels, hidden_dim, out_channels=3):
        # 调用父类初始化方法
        super(InvertedResidual, self).__init__()

        # 定义卷积序列：1x1升维 -> BN -> ReLU6 -> 1x1降维 -> BN -> ReLU
        self.conv = nn.Sequential(
            # 1x1卷积，用于通道升维（或保持不变）
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            # 批归一化，加速训练并提高稳定性
            nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
            # ReLU6激活函数，max(0, min(6, x))，限制激活值范围
            nn.ReLU6(inplace=True),

            # 注释掉的深度可分离卷积部分（Depthwise Convolution）
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),

            # 1x1逐点卷积（Pointwise Convolution），用于通道降维
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # 输出特征的批归一化
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            # ReLU激活函数
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播过程，直接通过定义的卷积序列
        return self.conv(x)


class BasicBlock(nn.Module):
    """基础残差块，用于ResNet-18和ResNet-34，包含两个3x3卷积层"""

    # 扩展因子，用于计算输出通道数（BasicBlock的输出通道是输入通道的1倍）
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 调用父类初始化方法
        super(BasicBlock, self).__init__()
        # 第一个3x3卷积层，可能改变特征图尺寸（通过stride）
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 第一个卷积层后的批归一化
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第二个3x3卷积层，通道数保持不变
        self.conv2 = conv3x3(planes, planes)
        # 第二个卷积层后的批归一化
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 下采样模块，用于匹配主分支和残差分支的尺寸和通道数
        self.downsample = downsample
        # 步长参数
        self.stride = stride

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x

        # 主分支：第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 主分支：第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样模块，则对残差进行下采样处理
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接：将主分支输出与残差相加
        out += residual
        # 最后通过ReLU激活函数
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """瓶颈残差块，用于ResNet-50/101/152，包含1x1、3x3、1x1三个卷积层"""

    # 扩展因子，输出通道数是中间通道数的4倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 调用父类初始化方法
        super(Bottleneck, self).__init__()
        # 第一个1x1卷积层，用于降维（减少计算量）
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # 第一个卷积层后的批归一化
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 第二个3x3卷积层，进行空间特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第二个卷积层后的批归一化
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 第三个1x1卷积层，用于升维（恢复通道数）
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # 第三个卷积层后的批归一化
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 下采样模块
        self.downsample = downsample
        # 步长参数
        self.stride = stride

    def forward(self, x):
        # 保存输入作为残差连接
        residual = x

        # 主分支：第一个1x1卷积块（降维）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 主分支：第二个3x3卷积块（特征提取）
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 主分支：第三个1x1卷积块（升维）
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果存在下采样模块，则对残差进行下采样处理
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接：将主分支输出与残差相加  如果没有下采样那么就是上一个输入的值，如果有那么就是将上个输入进行下采样后的值
        out += residual
        # 最后通过ReLU激活函数
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet主干网络实现类"""

    def __init__(self, block, layers):
        # 调用父类初始化方法
        super(ResNet, self).__init__()
        # 初始化输入通道数为64
        self.inplanes = 64

        # 第一个7x7卷积层，用于初始特征提取，输出通道64，步长2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 第一个卷积层后的批归一化
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，进一步下采样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建四个残差层  这里的每个block 都是一个Bottleneck 模型
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建残差层，包含多个残差块"""
        # 初始化下采样模块为None
        downsample = None
        # 当步长不为1或输入通道数不等于输出通道数时，需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建下采样序列：1x1卷积 + 批归一化
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        # 创建残差块列表
        layers = []
        # 添加第一个残差块（可能包含下采样）
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新输入通道数
        self.inplanes = planes * block.expansion
        # 添加剩余的残差块
        for i in range(1, blocks):  # blocks 代表的是有几层 这里self.inplanes 和 planes的维度是一样的吗
            layers.append(block(self.inplanes, planes))
        # 将列表封装为Sequential模块
        return nn.Sequential(*layers)

    def forward(self, input_x):
        """前向传播函数"""
        # 存储输出特征的字典
        out = {}
        # 初始卷积层
        x = self.conv1(input_x)
        x = self.bn1(x)
        x = self.relu(x)
        # 最大池化
        feature1 = self.maxpool(x)

        # 第一个残差层
        feature2 = self.layer1(feature1)
        out['res2'] = feature2  # 输出特征保存到字典中

        # 第二个残差层
        feature3 = self.layer2(feature2)
        out['res3'] = feature3  # 输出特征保存到字典中

        # 第三个残差层
        feature4 = self.layer3(feature3)
        out['res4'] = feature4  # 输出特征保存到字典中

        # 第四个残差层
        feature5 = self.layer4(feature4)
        out['res5'] = feature5  # 输出特征保存到字典中

        return out

    def init_weights(self, num_layers=50):
        """初始化网络权重，加载预训练模型"""
        # 注释掉的从网络加载预训练模型的方式
        # url = model_urls['resnet{}'.format(num_layers)]
        # pretrained_state_dict = model_zoo.load_url(url, model_dir='/home/code/pytorch_model/')
        # print('=> loading pretrained model {}'.format(url))
        # 从本地路径加载预训练模型
        pertained_model = r'/home/code/pytorch_model/resnet50-19c8e357.pth'
        pretrained_state_dict = torch.load(pertained_model)

        # 加载预训练权重，strict=False允许部分匹配
        self.load_state_dict(pretrained_state_dict, strict=False)


# ResNet各版本的配置字典：网络类型:(残差块类型, 各层残差块数量)
resnet_spec = {'resnet18': (BasicBlock, [2, 2, 2, 2]),
               'resnet34': (BasicBlock, [3, 4, 6, 3]),
               'resnet50': (Bottleneck, [3, 4, 6, 3]),
               'resnet101': (Bottleneck, [3, 4, 23, 3]),
               'resnet152': (Bottleneck, [3, 8, 36, 3])}