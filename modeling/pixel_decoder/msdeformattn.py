#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   msdeformattn.py
@Time    :   2022/10/02 16:51:09
@Author  :   BQH
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   修改自Mask2former,移除detectron2依赖
'''

# here put the import lib

# 导入数值计算库numpy
import numpy as np
# 导入fvcore库中的权重初始化模块
import fvcore.nn.weight_init as weight_init
# 导入PyTorch核心库
import torch
# 导入PyTorch神经网络模块
from torch import nn
# 导入PyTorch神经网络函数模块
from torch.nn import functional as F

# 从相对路径导入位置编码模块
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
# 从相对路径导入Transformer相关模块
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
# 从相对路径导入可变形注意力模块
from .ops.modules import MSDeformAttn


# MSDeformAttn Transformer编码器层实现
class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        # 调用父类构造函数
        super().__init__()

        # self attention - 创建可变形注意力机制
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # 创建dropout层用于防止过拟合
        self.dropout1 = nn.Dropout(dropout)
        # 创建LayerNorm层用于标准化
        self.norm1 = nn.LayerNorm(d_model)

        # ffn - 前馈神经网络部分
        self.linear1 = nn.Linear(d_model, d_ffn)  # 第一个线性层，扩展特征维度
        self.activation = _get_activation_fn(activation)  # 获取激活函数
        self.dropout2 = nn.Dropout(dropout)  # Dropout层
        self.linear2 = nn.Linear(d_ffn, d_model)  # 第二个线性层，恢复特征维度
        self.dropout3 = nn.Dropout(dropout)  # Dropout层
        self.norm2 = nn.LayerNorm(d_model)  # LayerNorm层

    @staticmethod
    def with_pos_embed(tensor, pos):
        # 静态方法：如果位置编码存在则与张量相加，否则直接返回张量
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        # 前馈神经网络前向传播
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))  # 线性->激活->dropout->线性
        src = src + self.dropout3(src2)  # 残差连接
        src = self.norm2(src)  # LayerNorm标准化
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention - 自注意力机制前向传播
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # LayerNorm标准化

        # ffn - 前馈神经网络
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        # 调用父类构造函数
        super().__init__()
        # 克隆多个编码器层
        self.layers = _get_clones(encoder_layer, num_layers)
        # 保存编码器层数量
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # 静态方法：获取参考点坐标
        reference_points_list = []  # 存储参考点列表
        # 遍历每个特征层的尺寸
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 生成网格坐标
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # 归一化y坐标
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # 归一化x坐标
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # 组合坐标
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H_ * W_, 2]
            reference_points_list.append(ref)
        # 拼接所有参考点
        reference_points = torch.cat(reference_points_list, 1)
        # 乘以有效比例
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        # 编码器前向传播
        output = src  # 初始化输出为输入
        # 获取参考点
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        # 遍历每一层编码器
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 ):
        # 调用父类构造函数
        super().__init__()

        # 保存模型维度和头数
        self.d_model = d_model
        self.nhead = nhead

        # 创建编码器层
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        # 创建编码器
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        # 创建层级嵌入参数 与后面的位置编码进行相加
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # 重置参数
        self._reset_parameters()

    def _reset_parameters(self):
        # 重置模型参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 对维度大于1的参数使用Xavier初始化
        # 对模块进行初始化
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()  # 对可变形注意力模块重置参数
        nn.init.normal_(self.level_embed)  # 对层级嵌入使用正态分布初始化

    def get_valid_ratio(self, mask):
        # 获取有效比例
        _, H, W = mask.shape  # 获取mask形状
        valid_H = torch.sum(~mask[:, :, 0], 1)  # 计算有效高度
        valid_W = torch.sum(~mask[:, 0, :], 1)  # 计算有效宽度
        valid_ratio_h = valid_H.float() / H  # 计算高度有效比例
        valid_ratio_w = valid_W.float() / W  # 计算宽度有效比例
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # 组合比例
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        # 创建mask 这个mask不考虑维度维
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder - 为编码器准备输入
        src_flatten = []  # 扁平化的源特征
        mask_flatten = []  # 扁平化的mask
        lvl_pos_embed_flatten = []  # 扁平化的位置嵌入
        spatial_shapes = []  # 空间形状
        # 遍历源特征、mask和位置嵌入
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape  # 获取批次大小、通道数、高、宽
            spatial_shape = (h, w)  # 空间形状
            spatial_shapes.append(spatial_shape)  # 添加到空间形状列表
            src = src.flatten(2).transpose(1, 2)  # 扁平化空间维度并转置 也就是h和w相乘 然后将维度放到最后
            mask = mask.flatten(1)  # 扁平化mask h和w相乘
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # 扁平化位置嵌入并转置
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # 添加层级嵌入
            lvl_pos_embed_flatten.append(lvl_pos_embed)  # 添加到扁平化位置嵌入列表
            src_flatten.append(src)  # 添加到扁平化源特征列表
            mask_flatten.append(mask)  # 添加到扁平化mask列表
        # 拼接所有特征
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # 层级位置编码
        # 转换空间形状为张量
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 计算层级起始索引
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 计算有效比例
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder - 编码器处理
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
            self,
            input_shape,
            transformer_dropout=0.1,
            transformer_nheads=8,
            transformer_dim_feedforward=2048,
            transformer_enc_layers=6,
            conv_dim=256,
            mask_dim=256,

            # deformable transformer encoder args
            transformer_in_features=["res3", "res4", "res5"],
            common_stride=4,
    ):
        # 调用父类构造函数
        super().__init__()
        # backbone中["res3", "res4", "res5"]特征层的(channel, stride), eg. [(32,4), (64, 8),(128, 16),(256, 32)]
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}

        # this is the input shape of pixel decoder
        self.in_features = [k for k, v in input_shape.items()]  # starting from "res3" to "res5"
        self.feature_channels = [v.channel for k, v in input_shape.items()]  # eg. [16, 64, 128, 256]

        # this is the input shape of transformer encoder (could use less features than pixel decoder
        self.transformer_in_features = [k for k, v in transformer_input_shape.items()]  # starting from "res3" to "res5"
        transformer_in_channels = [v.channel for k, v in transformer_input_shape.items()]  # eg. [64, 128, 256]
        self.transformer_feature_strides = [v.stride for k, v in
                                            transformer_input_shape.items()]  # to decide extra FPN layers

        # 计算特征层级数量
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        # 如果特征层级大于1
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []  # 输入投影列表
            # from low resolution to high resolution (res5 -> res3) - 从低分辨率到高分辨率
            for in_channels in transformer_in_channels[::-1]:  # 反向遍历通道数
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),  # 1x1卷积
                    nn.GroupNorm(32, conv_dim),  # 组归一化
                ))
            self.input_proj = nn.ModuleList(input_proj_list)  # 创建模块列表
        else:
            # 特征层级等于1的情况
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),  # 1x1卷积
                    nn.GroupNorm(32, conv_dim),  # 组归一化
                )])

        # 初始化投影层权重和偏置
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 创建变形注意力Transformer编码器
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        # 计算位置编码步数
        N_steps = conv_dim // 2
        # 创建位置编码层
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # 保存mask维度
        self.mask_dim = mask_dim
        # use 1x1 conv instead - 使用1x1卷积
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # 初始化mask特征卷积权重
        weight_init.c2_xavier_fill(self.mask_features)

        # 设置maskformer特征层级数量
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        # 设置公共步长
        self.common_stride = common_stride

        # extra fpn levels - 额外FPN层级
        stride = min(self.transformer_feature_strides)  # 获取最小步长
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))  # 计算FPN层级数

        # 创建横向连接和输出卷积
        lateral_convs = []
        output_convs = []

        # 遍历特征通道创建FPN层 这个和后面前向传播中的内容是首尾呼应
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):  # res2 -> fpn
            # 横向连接卷积 1*1 卷积 将维度上升/下降到conv_dim 维度
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                                         nn.GroupNorm(32, conv_dim),
                                         nn.ReLU(inplace=True))

            # 输出卷积 3*3 卷积
            output_conv = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
                                        nn.GroupNorm(32, conv_dim),  # num_groups, num_channels
                                        nn.ReLU(inplace=True))

            # 初始化权重
            weight_init.c2_xavier_fill(lateral_conv[0])
            weight_init.c2_xavier_fill(output_conv[0])
            # 添加模块
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        # 将卷积按从低到高分辨率的顺序排列
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward_features(self, features):
        srcs = []  # 源特征列表
        pos = []  # 位置编码列表
        # Reverse feature maps into top-down order (from low to high resolution), 'res5' -> 'res3'
        # 将特征图按从低到高分辨率的顺序排列 通道数由高到低
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision - 变形注意力不支持半精度
            srcs.append(self.input_proj[idx](x))  # 应用输入投影 这里的投影层的输入不一样但是输出层都转变为128维度
            pos.append(self.pe_layer(x))  # 生成位置编码

        # Transformer编码器处理
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]  # 获取批次大小

        # 计算分割大小 计算每层原本的大小
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        # 分割特征 从中间维度进行划分 也就是展平的特征维度进行划分
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []  # 输出特征列表
        multi_scale_features = []  # 多尺度特征列表
        num_cur_levels = 0  # 当前层级计数
        # 遍历分割后的特征 以下步骤是为了还原特征图的大小
        for i, z in enumerate(y):
            # 转置并reshape特征
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        # 添加额外的FPN层级，按从低到高分辨率顺序
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):  # 这里是从所有的层级开始算起
            x = features[f].float()  # 获取特征 对应的特征层的值
            lateral_conv = self.lateral_convs[idx]  # 获取横向连接卷积
            output_conv = self.output_convs[idx]  # 获取输出卷积
            cur_fpn = lateral_conv(x)  # 应用横向连接
            # Following FPN implementation, we use nearest upsampling here
            # 按照FPN实现，这里使用最近邻上采样  将多头可变tranformer的输出的最后一层（特征图更大的 更能区分细节）上采样后和最后一层的上采样过后的特征图进行相加的操作
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)  # 应用输出卷积 进行3*3 的卷积 约等于全连接
            out.append(y)  # 添加到输出列表

        # 收集多尺度特征
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        # 后面的步骤很多是还原操作
        # 返回mask特征、输出特征和多尺度特征
        return self.mask_features(out[-1]), out[0], multi_scale_features