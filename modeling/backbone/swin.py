# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py

import numpy as np
# 导入numpy库，用于数值计算
from addict import Dict
# 导入Dict类，提供字典增强功能
import torch
# 导入PyTorch核心库
import torch.nn as nn
# 导入PyTorch神经网络模块
import torch.nn.functional as F
# 导入PyTorch函数式接口
import torch.utils.checkpoint as checkpoint
# 导入检查点机制，用于节省内存
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# 从timm库导入相关层函数：DropPath用于随机深度，to_2tuple用于转换为二元组，trunc_normal_用于截断正态分布初始化

# from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
# 注释掉detectron2相关导入


class Mlp(nn.Module):
    # 定义多层感知机(Multilayer Perceptron)类，继承自nn.Module
    """Multilayer perceptron."""

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        # 初始化函数，定义MLP的结构参数
        # in_features: 输入特征维度
        # hidden_features: 隐藏层特征维度，默认与输入相同
        # out_features: 输出特征维度，默认与输入相同
        # act_layer: 激活函数层，默认为GELU
        # drop: dropout比率，默认为0
        super().__init__()
        # 调用父类初始化方法
        out_features = out_features or in_features
        # 如果未指定输出特征维度，则与输入特征维度相同
        hidden_features = hidden_features or in_features
        # 如果未指定隐藏层特征维度，则与输入特征维度相同
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 定义第一个全连接层
        self.act = act_layer()
        # 定义激活函数层
        self.fc2 = nn.Linear(hidden_features, out_features)
        # 定义第二个全连接层
        self.drop = nn.Dropout(drop)
        # 定义dropout层

    def forward(self, x):
        # 前向传播函数
        # x: 输入张量
        x = self.fc1(x)
        # 通过第一个全连接层
        x = self.act(x)
        # 通过激活函数
        x = self.drop(x)
        # 应用dropout
        x = self.fc2(x)
        # 通过第二个全连接层
        x = self.drop(x)
        # 再次应用dropout
        return x
        # 返回输出结果


def window_partition(x, window_size):
    # 窗口划分函数，将特征图划分为不重叠的窗口 只是做形状的优化
    # x: 输入特征图，形状为(B, H, W, C)
    # window_size: 窗口大小
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 获取输入张量的形状信息：批量大小、高、宽、通道数
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重新组织张量形状，为窗口划分做准备
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 调整维度顺序并重新整形，得到窗口化的特征
    return windows
    # 返回窗口化后的特征


def window_reverse(windows, window_size, H, W):
    # 窗口逆变换函数，将窗口化的特征还原为原始特征图 窗口还原从适合transformer的格式 转换b,h,w,c
    # windows: 窗口化特征
    # window_size: 窗口大小
    # H, W: 原始特征图的高度和宽度
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 计算批量大小
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 重新组织窗口特征的形状
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # 调整维度顺序并重新整形，还原为原始特征图
    return x
    # 返回还原后的特征图


class WindowAttention(nn.Module):
    # 窗口注意力机制类，实现基于窗口的多头自注意力
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        # 初始化窗口注意力模块
        # dim: 输入通道数
        # window_size: 窗口大小(高,宽)
        # num_heads: 注意力头数
        # qkv_bias: 是否为query,key,value添加偏置
        # qk_scale: qk缩放因子
        # attn_drop: 注意力权重的dropout比率
        # proj_drop: 输出的dropout比率

        super().__init__()
        # 调用父类初始化方法
        self.dim = dim
        # 保存输入通道数
        self.window_size = window_size  # Wh, Ww
        # 保存窗口大小
        self.num_heads = num_heads
        # 保存注意力头数
        head_dim = dim // num_heads
        # 计算每个注意力头的维度
        self.scale = head_dim ** -0.5 if qk_scale is None else qk_scale
        # 设置缩放因子

        # define a parameter table of relative position bias
        # 定义相对位置偏置参数表 前面两个维度代表相对的位置  后面一个代表头的个数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # 创建相对位置偏置表参数

        # get pair-wise relative position index for each token inside the window
        # 获取窗口内每个token的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        # 生成窗口高度方向的坐标
        coords_w = torch.arange(self.window_size[1])
        # 生成窗口宽度方向的坐标
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # 创建网格坐标
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 将坐标展平
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # 计算相对坐标
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 调整相对坐标维度顺序
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # 调整高度方向相对坐标
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 调整宽度方向相对坐标
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # 缩放高度方向相对坐标
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # 计算相对位置索引
        self.register_buffer("relative_position_index", relative_position_index)
        # 注册相对位置索引为缓冲区
        # 线性变换层次 生成 qkv  这里是self attn 这里乘以3是因为是3头还是因为是 qkv 这三个内容  答案是后者
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 定义线性变换层，用于生成query, key, value
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义注意力dropout层
        self.proj = nn.Linear(dim, dim)
        # 定义输出投影层
        self.proj_drop = nn.Dropout(proj_drop)
        # 定义输出dropout层

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        # 对相对位置偏置表进行截断正态分布初始化
        self.softmax = nn.Softmax(dim=-1)
        # 定义softmax层

    def forward(self, x, mask=None):
        # 前向传播函数
        # x: 输入特征，形状为(num_windows*B, N, C)
        # mask: 注意力掩码，形状为(num_windows, Wh*Ww, Wh*Ww)或None
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # 获取输入张量的形状信息 在这个转化中就存在在通道维度上分头操作
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # 通过线性层生成qkv并向量重排
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 分离query, key, value
        # 点积缩放的注意力
        q = q * self.scale
        # 对query进行缩放
        attn = q @ k.transpose(-2, -1)
        # 计算注意力分数

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        # 获取相对位置偏置 可以学习
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 调整相对位置偏置维度顺序
        attn = attn + relative_position_bias.unsqueeze(0)
        # 将相对位置偏置加入注意力分数

        if mask is not None:
            # 如果存在掩码
            nW = mask.shape[0]
            # 获取窗口数量
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # 应用掩码
            attn = attn.view(-1, self.num_heads, N, N)
            # 重新整形
            attn = self.softmax(attn)
            # 应用softmax
        else:
            # 如果不存在掩码
            attn = self.softmax(attn)
            # 直接应用softmax

        attn = self.attn_drop(attn)
        # 应用注意力dropout

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 计算加权和并重新整形
        x = self.proj(x)
        # 通过输出投影层
        x = self.proj_drop(x)
        # 应用输出dropout
        return x
        # 返回输出结果


class SwinTransformerBlock(nn.Module):
    # Swin Transformer基本块类
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        # 初始化Swin Transformer块
        # dim: 输入通道数
        # num_heads: 注意力头数
        # window_size: 窗口大小
        # shift_size: 移位大小，用于SW-MSA
        # mlp_ratio: MLP隐藏维度与嵌入维度的比率
        # qkv_bias: 是否为qkv添加偏置
        # qk_scale: qk缩放因子
        # drop: dropout比率
        # attn_drop: 注意力dropout比率
        # drop_path: 随机深度比率
        # act_layer: 激活函数层
        # norm_layer: 归一化层
        super().__init__()
        # 调用父类初始化方法
        self.dim = dim
        # 保存输入通道数
        self.num_heads = num_heads
        # 保存注意力头数
        self.window_size = window_size
        # 保存窗口大小
        self.shift_size = shift_size
        # 保存移位大小
        self.mlp_ratio = mlp_ratio
        # 保存MLP比率
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # 验证移位大小的有效性

        self.norm1 = norm_layer(dim)
        # 定义第一个归一化层
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # 定义窗口注意力模块

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 定义随机深度层
        self.norm2 = norm_layer(dim)
        # 定义第二个归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 计算MLP隐藏层维度
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        # 定义MLP模块

        self.H = None
        # 初始化高度占位符
        self.W = None
        # 初始化宽度占位符

    def forward(self, x, mask_matrix):
        # 前向传播函数
        # x: 输入特征，张量大小(B, H*W, C)
        # mask_matrix: 用于循环移位的注意力掩码
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        # 获取输入张量的形状信息
        H, W = self.H, self.W
        # 获取空间分辨率
        assert L == H * W, "input feature has wrong size"
        # 验证输入特征大小正确性

        shortcut = x
        # 保存输入作为捷径连接
        x = self.norm1(x)
        # 通过第一个归一化层
        x = x.view(B, H, W, C)
        # 重新整形为4D张量

        # pad feature maps to multiples of window size
        # 填充特征图使其大小为窗口大小的倍数
        pad_l = pad_t = 0
        # 初始化左和上填充大小
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        # 计算右填充大小
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        # 计算下填充大小
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # 对特征图进行填充
        _, Hp, Wp, _ = x.shape
        # 获取填充后的形状

        # cyclic shift
        # 循环移位
        if self.shift_size > 0:
            # 如果移位大小大于0 如果方向为负那么是 向上 向左
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 进行循环移位
            attn_mask = mask_matrix
            # 使用提供的掩码
        else:
            # 如果不需要移位
            shifted_x = x
            # 不进行移位
            attn_mask = None
            # 不使用掩码

        # partition windows
        # 窗口划分
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        # 将移位后的特征划分为窗口
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        # 重新整形窗口特征

        # W-MSA/SW-MSA
        # 窗口多头自注意力/移位窗口多头自注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        # 通过窗口注意力模块

        # merge windows
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # 重新整形注意力窗口
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        # 将窗口合并回原始特征图

        # reverse cyclic shift
        # 逆循环移位
        if self.shift_size > 0:
            # 如果之前进行了移位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # 进行逆向移位
        else:
            # 如果之前未进行移位
            x = shifted_x
            # 直接使用移位后的特征

        if pad_r > 0 or pad_b > 0:
            # 如果进行了填充
            x = x[:, :H, :W, :].contiguous()
            # 移除填充部分

        x = x.view(B, H * W, C)
        # 重新整形为3D张量

        # FFN
        # 残差连接 加上
        x = shortcut + self.drop_path(x)
        # 捷径连接和随机深度  dropout +mlp两个线性 +norm2 之后再进行残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # 第二个捷径连接和MLP

        return x
        # 返回输出结果


class PatchMerging(nn.Module):
    # Patch合并层类
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        # 初始化Patch合并层
        # dim: 输入通道数
        # norm_layer: 归一化层
        super().__init__()
        # 调用父类初始化方法
        self.dim = dim
        # 保存输入通道数 线性降维
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 定义线性变换层，用于降维
        self.norm = norm_layer(4 * dim)
        # 定义归一化层

    def forward(self, x, H, W):
        # 前向传播函数
        # x: 输入特征，张量大小(B, H*W, C)
        # H, W: 输入特征的空间分辨率
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        # 获取输入张量的形状信息
        assert L == H * W, "input feature has wrong size"
        # 验证输入特征大小正确性

        x = x.view(B, H, W, C)
        # 重新整形为4D张量

        # padding
        # 填充
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        # 检查是否需要填充
        if pad_input:
            # 如果需要填充
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            # 进行填充

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # 获取偶数行偶数列的特征
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # 获取奇数行偶数列的特征
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # 获取偶数行奇数列的特征
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # 获取奇数行奇数列的特征
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # 拼接四个特征
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        # 重新整形

        x = self.norm(x)
        # 通过归一化层
        x = self.reduction(x)
        # 通过降维线性层

        return x
        # 返回输出结果


class BasicLayer(nn.Module):
    # Swin Transformer基本层类，代表一个阶段
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
    ):
        # 初始化基本层
        # dim: 特征通道数
        # depth: 该阶段的深度(块的数量)
        # num_heads: 注意力头数
        # window_size: 局部窗口大小
        # mlp_ratio: MLP隐藏维度与嵌入维度的比率
        # qkv_bias: 是否为qkv添加偏置
        # qk_scale: qk缩放因子
        # drop: dropout比率
        # attn_drop: 注意力dropout比率
        # drop_path: 随机深度比率
        # norm_layer: 归一化层
        # downsample: 下采样层
        # use_checkpoint: 是否使用检查点节省内存
        super().__init__()
        # 调用父类初始化方法
        self.window_size = window_size
        # 保存窗口大小
        self.shift_size = window_size // 2
        # 计算移位大小
        self.depth = depth
        # 保存深度
        self.use_checkpoint = use_checkpoint
        # 保存检查点使用标志

        # build blocks
        # 构建块
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # 创建Swin Transformer块列表

        # patch merging layer
        # Patch合并层
        if downsample is not None:
            # 如果指定了下采样层
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
            # 创建下采样层
        else:
            # 如果未指定下采样层
            self.downsample = None
            # 不使用下采样

    def forward(self, x, H, W):
        # 前向传播函数
        # x: 输入特征，张量大小(B, H*W, C)
        # H, W: 输入特征的空间分辨率
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        # 为SW-MSA计算注意力掩码
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        # 计算填充后的高度 上面
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 计算填充后的宽度 上面
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        # 创建图像掩码 创建切片
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        # 定义高度切片
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        # 定义宽度切片
        cnt = 0
        # 初始化计数器 这里是划分为9个区域 也就是将值划分成9个区域 区分块
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # 为掩码分配值
        # 上面两个 for循环 是否可以用以下内容进行替代 x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        # 划分掩码窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 重新整形掩码窗口
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 计算注意力掩码 -100 这个惩罚足够大 ，使得softmax时候基本上为0
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        # 填充注意力掩码
        # 这里是两个步骤 一步W_MAS 一步SW_MSA
        for blk in self.blocks:
            # 遍历所有块
            blk.H, blk.W = H, W
            # 设置块的高度和宽度
            if self.use_checkpoint:
                # 如果使用检查点
                x = checkpoint.checkpoint(blk, x, attn_mask)
                # 使用检查点机制执行块
            else:
                # 如果不使用检查点
                x = blk(x, attn_mask)
                # 直接执行块
        if self.downsample is not None:
            # 如果存在下采样层
            x_down = self.downsample(x, H, W)
            # 执行下采样
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            # 计算下采样后的空间分辨率
            return x, H, W, x_down, Wh, Ww
            # 返回结果
        else:
            # 如果不存在下采样层
            return x, H, W, x, H, W
            # 返回结果


class PatchEmbed(nn.Module):
    # 图像到Patch嵌入类
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        # 初始化Patch嵌入层
        # patch_size: Patch标记大小
        # in_chans: 输入图像通道数
        # embed_dim: 线性投影输出通道数
        # norm_layer: 归一化层
        super().__init__()
        # 调用父类初始化方法
        patch_size = to_2tuple(patch_size)
        # 转换patch_size为二元组
        self.patch_size = patch_size
        # 保存patch_size

        self.in_chans = in_chans
        # 保存输入通道数
        self.embed_dim = embed_dim
        # 保存嵌入维度
        # 当前是一个卷积操作 作用是将 输入的通道转化为embed的维度
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 定义投影卷积层
        if norm_layer is not None:
            # 如果指定了归一化层
            self.norm = norm_layer(embed_dim)
            # 创建归一化层
        else:
            # 如果未指定归一化层
            self.norm = None
            # 不使用归一化

    def forward(self, x):
        # 前向传播函数
        """Forward function."""
        # padding
        # 填充
        _, _, H, W = x.size()
        # 获取输入张量的空间尺寸
        if W % self.patch_size[1] != 0:  # 宽度方向的填充
            # 如果宽度不能被patch_size整除 为什么这里要减去一个W
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
            # 在宽度方向进行填充
        if H % self.patch_size[0] != 0:  # 高度方向的填充
            # 如果高度不能被patch_size整除
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            # 在高度方向进行填充
        # 将通道维度转化 然后对高和宽进行下采样操作 都除以2
        x = self.proj(x)  # B C Wh Ww
        # 通过投影卷积层
        if self.norm is not None:
            # 如果存在归一化层
            Wh, Ww = x.size(2), x.size(3)
            # 获取特征图尺寸
            x = x.flatten(2).transpose(1, 2)
            # 展平并转置
            x = self.norm(x)
            # 通过归一化层
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
            # 转置并重新整形
        # 最后将值进行还原
        return x
        # 返回输出结果


class SwinTransformer(nn.Module):
    # Swin Transformer主干网络类
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            pretrain_img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
    ):
        # 初始化Swin Transformer
        # pretrain_img_size: 预训练模型的输入图像大小
        # patch_size: Patch大小
        # in_chans: 输入图像通道数
        # embed_dim: 嵌入维度
        # depths: 每个阶段的深度
        # num_heads: 每个阶段的注意力头数
        # window_size: 窗口大小
        # mlp_ratio: MLP比率
        # qkv_bias: qkv偏置
        # qk_scale: qk缩放因子
        # drop_rate: dropout比率
        # attn_drop_rate: 注意力dropout比率
        # drop_path_rate: 随机深度比率
        # norm_layer: 归一化层
        # ape: 是否使用绝对位置嵌入
        # patch_norm: 是否在Patch嵌入后使用归一化
        # out_indices: 输出的阶段索引
        # frozen_stages: 冻结的阶段数
        # use_checkpoint: 是否使用检查点
        super().__init__()
        # 调用父类初始化方法

        self.pretrain_img_size = pretrain_img_size
        # 保存预训练图像大小
        self.num_layers = len(depths)
        # 计算层数
        self.embed_dim = embed_dim
        # 保存嵌入维度
        self.ape = ape
        # 保存绝对位置嵌入标志
        self.patch_norm = patch_norm
        # 保存Patch归一化标志
        self.out_indices = out_indices
        # 保存输出索引
        self.frozen_stages = frozen_stages
        # 保存冻结阶段数

        # split image into non-overlapping patches
        # 将图像分割为不重叠的patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        # 创建Patch嵌入层

        # absolute position embedding
        # 绝对位置嵌入
        if self.ape:
            # 如果使用绝对位置嵌入
            pretrain_img_size = to_2tuple(pretrain_img_size)
            # 转换预训练图像大小为二元组
            patch_size = to_2tuple(patch_size)
            # 转换patch大小为二元组
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]
            # 计算patches分辨率

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            # 创建绝对位置嵌入参数 把张量里的值按 N(μ, σ²) 分布重新填充，同时把超出 [μ−2σ, μ+2σ] 的尾巴直接砍掉，
            # 既保证均值方差，又防止极端值出现。
            trunc_normal_(self.absolute_pos_embed, std=0.02)
            # 对绝对位置嵌入进行初始化

        self.pos_drop = nn.Dropout(p=drop_rate)
        # 创建位置dropout层

        # stochastic depth
        # 随机深度
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
            # torch.linspace 用来 在指定区间 [start, end] 内等间隔地生成 固定数量 的一维张量。
        ]  # stochastic depth decay rule
        # 计算随机深度衰减规则

        # build layers
        # 构建层
        self.layers = nn.ModuleList()
        # 创建层列表
        for i_layer in range(self.num_layers):
            # 遍历每一层
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            # 创建基本层
            self.layers.append(layer)
            # 添加到层列表

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # 计算每层的特征数
        self.num_features = num_features
        # 保存特征数

        # add a norm layer for each output
        # 为每个输出添加归一化层
        for i_layer in out_indices:
            # 遍历输出索引
            layer = norm_layer(num_features[i_layer])
            # 创建归一化层
            layer_name = f"norm{i_layer}"
            # 生成层名称
            self.add_module(layer_name, layer)
            # 添加模块

        self._freeze_stages()
        # 冻结指定阶段

    def _freeze_stages(self):
        # 冻结阶段函数
        if self.frozen_stages >= 0:
            # 如果需要冻结Patch嵌入阶段
            self.patch_embed.eval()
            # 设置为评估模式
            for param in self.patch_embed.parameters():
                param.requires_grad = False
                # 冻结参数

        if self.frozen_stages >= 1 and self.ape:
            # 如果需要冻结绝对位置嵌入
            self.absolute_pos_embed.requires_grad = False
            # 冻结绝对位置嵌入参数

        if self.frozen_stages >= 2:
            # 如果需要冻结更多阶段
            self.pos_drop.eval()
            # 设置位置dropout为评估模式
            for i in range(0, self.frozen_stages - 1):
                # 遍历需要冻结的阶段
                m = self.layers[i]
                # 获取层
                m.eval()
                # 设置为评估模式
                for param in m.parameters():
                    param.requires_grad = False
                    # 冻结参数

    def init_weights(self, pretrained=None):
        # 权重初始化函数
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            # 权重初始化辅助函数
            if isinstance(m, nn.Linear):
                # 如果是线性层
                trunc_normal_(m.weight, std=0.02)
                # 对权重进行截断正态分布初始化
                if isinstance(m, nn.Linear) and m.bias is not None:
                    # 如果存在偏置
                    nn.init.constant_(m.bias, 0)
                    # 将偏置初始化为0
            elif isinstance(m, nn.LayerNorm):
                # 如果是LayerNorm层
                nn.init.constant_(m.bias, 0)
                # 将偏置初始化为0
                nn.init.constant_(m.weight, 1.0)
                # 将权重初始化为1

    def forward(self, x):
        # 前向传播函数
        """Forward function."""
        x = self.patch_embed(x)
        # 通过Patch嵌入层

        Wh, Ww = x.size(2), x.size(3)
        # 获取特征图尺寸
        if self.ape:
            # 如果使用绝对位置嵌入
            # interpolate the position embedding to the corresponding size
            # 将位置嵌入插值到相应大小
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            # 插值绝对位置嵌入
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            # 添加位置嵌入并重新整形
        else:
            # 如果不使用绝对位置嵌入
            x = x.flatten(2).transpose(1, 2)
            # 直接重新整形
        x = self.pos_drop(x)
        # 应用位置dropout

        outs = {}
        # 初始化输出字典
        for i in range(self.num_layers):
            # 遍历每一层
            layer = self.layers[i]
            # 获取层
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # 执行层操作

            if i in self.out_indices:
                # 如果该层需要输出
                norm_layer = getattr(self, f"norm{i}")
                # 获取对应归一化层
                x_out = norm_layer(x_out)
                # 应用归一化

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                # 重新整形输出
                outs["res{}".format(i + 2)] = out
                # 保存输出

        return outs
        # 返回输出结果

    def train(self, mode=True):
        # 训练模式设置函数
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        # 调用父类训练模式设置
        self._freeze_stages()
        # 保持指定层冻结


class D2SwinTransformer(SwinTransformer):
    # Detectron2兼容的Swin Transformer类
    def __init__(self, cfg):
        # 初始化函数，接受配置参数

        pretrain_img_size = cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE
        # 获取预训练图像大小配置
        patch_size = cfg.MODEL.SWIN.PATCH_SIZE
        # 获取patch大小配置
        in_chans = 3
        # 设置输入通道数为3
        embed_dim = cfg.MODEL.SWIN.EMBED_DIM
        # 获取嵌入维度配置
        depths = cfg.MODEL.SWIN.DEPTHS
        # 获取深度配置
        num_heads = cfg.MODEL.SWIN.NUM_HEADS
        # 获取注意力头数配置
        window_size = cfg.MODEL.SWIN.WINDOW_SIZE
        # 获取窗口大小配置
        mlp_ratio = cfg.MODEL.SWIN.MLP_RATIO
        # 获取MLP比率配置
        qkv_bias = cfg.MODEL.SWIN.QKV_BIAS
        # 获取qkv偏置配置
        qk_scale = cfg.MODEL.SWIN.QK_SCALE
        # 获取qk缩放配置
        drop_rate = cfg.MODEL.SWIN.DROP_RATE
        # 获取dropout比率配置
        attn_drop_rate = cfg.MODEL.SWIN.ATTN_DROP_RATE
        # 获取注意力dropout比率配置
        drop_path_rate = cfg.MODEL.SWIN.DROP_PATH_RATE
        # 获取随机深度比率配置
        norm_layer = nn.LayerNorm
        # 设置归一化层
        ape = cfg.MODEL.SWIN.APE
        # 获取绝对位置嵌入配置
        patch_norm = cfg.MODEL.SWIN.PATCH_NORM
        # 获取Patch归一化配置
        use_checkpoint = cfg.MODEL.SWIN.USE_CHECKPOINT
        # 获取检查点使用配置

        super().__init__(
            pretrain_img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            use_checkpoint=use_checkpoint,
        )
        # 调用父类初始化

        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES
        # 设置输出特征

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        # 设置输出特征步幅
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }
        # 设置输出特征通道数

    def forward(self, x):
        # 前向传播函数
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
                x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        # 验证输入张量维度
        outputs = {}
        # 初始化输出字典 这里继承父类的SwinTransformer 使用这里的前向传播
        y = super().forward(x)
        # 调用父类前向传播
        for k in y.keys():
            # 遍历所有输出键
            if k in self._out_features:
                # 如果该键在需要输出的特征中
                outputs[k] = y[k]
                # 保存对应特征
        return outputs
        # 返回输出结果

    def output_shape(self):
        # 输出形状函数
        backbone_feature_shape = dict()
        # 初始化骨干网络特征形状字典
        for name in self._out_features:
            # 遍历所有输出特征名称
            backbone_feature_shape[name] = Dict(
                {'channel': self._out_feature_channels[name], 'stride': self._out_feature_strides[name]})
            # 设置对应特征的通道数和步幅
        return backbone_feature_shape
        # 返回特征形状字典

    @property
    def size_divisibility(self):
        # 大小可除性属性
        return 32
        # 返回32