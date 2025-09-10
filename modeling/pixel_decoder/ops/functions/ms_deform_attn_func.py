# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


# try:
#     import MultiScaleDeformableAttention as MSDA
# except ModuleNotFoundError as e:
#     info_string = (
#         "\n\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\n"
#         "\t`cd mask2former/modeling/pixel_decoder/ops`\n"
#         "\t`sh make.sh`\n"
#     )
#     raise ModuleNotFoundError(info_string)


# class MSDeformAttnFunction(Function):
#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         ctx.im2col_step = im2col_step
#         output = MSDA.ms_deform_attn_forward(
#             value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value, grad_sampling_loc, grad_attn_weight = \
#             MSDA.ms_deform_attn_backward(
#                 value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

#         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # 函数文档字符串，说明输入张量的维度信息
    # @value: bs, sum(h, w), num_head, dim (输入特征值张量，bs为batch size，sum(h,w)为所有特征图像素点总和，num_head为注意力头数，dim为每个头的维度)
    # @sampling_locations: bs, sum(h, w), num_head, num_layer, 4, 2 (采样位置，num_layer为层数，4为每个位置采样4个点，2为坐标xy)
    # @attention_weights: bs, sum(h, w), num_head, num_layer, 4 (注意力权重)

    # 获取输入value张量的维度信息
    # N_: batch size, S_: 所有特征图的像素点总数, M_: 注意力头数, Dim: 每个头的维度
    N_, S_, M_, Dim = value.shape

    # 获取采样位置张量的维度信息
    # Lq_: 查询位置数(等于S_), L_: 特征层数, P_: 每个位置的采样点数(等于4)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    # 根据每个特征图的尺寸将value分割成多个列表
    # value_spatial_shapes包含每个特征图的高和宽，[H_ * W_ for H_, W_ in value_spatial_shapes]计算每个特征图的像素点数  在第一个为维度上瓜分，也就是特征向量维度上进行瓜分
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # 将采样位置从[0,1]范围转换到[-1,1]范围，因为F.grid_sample要求grid的范围是[-1,1]
    sampling_grids = 2 * sampling_locations - 1

    # 初始化采样值列表
    sampling_value_list = []

    # 遍历每个特征图层级
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # 对当前层级的value进行维度变换，便于后续采样操作
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, H_, W_
        # 首先将todo 每个头的维度和头数合并,然后转置使通道维在前，最后reshape成适合grid_sample的格式
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, Dim, H_,
                                                                       W_)  # eg. [bs * 8, 32, 28, 28, 28]

        # 获取当前层级的采样网格，并进行维度变换 选择指定的层级在sampling_grids中
        # N_, Lq_, M_, P_, 3 -> N_, M_, Lq_, P_, 3 -> N_*M_, Lq_, P_, 3
        sampling_grid_l_ = sampling_grids[:, :, :, lid_]  # 这个时候对比原图来说 少了层级维度
        # 调整维度顺序并压平批次和头的维度
        sampling_grid_l_ = sampling_grid_l_.transpose(1, 2).flatten(0, 1)  # eg. [bs * 8, 1045, 3, 3]

        # 使用双线性插值在特征图上采样对应位置的值
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros',
                                          align_corners=False)  # eg. [bs * 8, 32, 1045, 4]
        # 将采样结果添加到列表中
        sampling_value_list.append(sampling_value_l_)

    # 对注意力权重进行维度变换，为后续加权计算做准备 按照头和批次进行聚合 然后将层级和点位进行聚合
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_,
                                                                  L_ * P_)  # eg. [bs * 8, 1, 1045, 4 * 4], 4个特征层 * 4个采样点

    # 将所有层级的采样值堆叠，并与注意力权重相乘后求和，得到最终输出  squeeze(2) 只会在“第 2 维（从 0 数）长度为 1 时才真正删掉这一维”。
    # torch.stack(sampling_value_list, dim=-2): [bs * 8, 32, 1045, 4, num_layer] -> [bs * 8, 32, 1045, 4 * 4], 4个特征层 * 4个采样点
    output = (torch.stack(sampling_value_list, dim=-2).squeeze(2).flatten(-2) * attention_weights).sum(-1).view(N_,
                                                                                                                M_ * Dim,
                                                                                                                Lq_)

    # 调整输出张量的维度顺序并确保内存连续
    return output.transpose(1, 2).contiguous()