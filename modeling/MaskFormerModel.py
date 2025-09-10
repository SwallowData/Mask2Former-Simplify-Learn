#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MaskFormerModel.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于DeformTransAtten的分割网络
'''

# here put the import lib
# 导入PyTorch神经网络模块
from torch import nn
# 导入addict库中的Dict类，用于创建嵌套字典
from addict import Dict

# 导入ResNet相关模块
from .backbone.resnet import ResNet, resnet_spec
# 导入Swin Transformer相关模块
from .backbone.swin import D2SwinTransformer
# 导入基于可变形注意力的像素解码器
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
# 导入多尺度掩码Transformer解码器
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


# 定义MaskFormer的头部网络类，继承自nn.Module
class MaskFormerHead(nn.Module):
    # 初始化函数，接收配置参数和输入特征形状
    def __init__(self, cfg, input_shape):
        # 调用父类初始化函数
        super().__init__()
        # 初始化像素解码器
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape)
        # 初始化预测器
        self.predictor = self.predictor_init(cfg)

    # 像素解码器初始化函数
    def pixel_decoder_init(self, cfg, input_shape):
        # 获取通用步长参数
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        # 获取Transformer dropout率
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        # 获取Transformer注意力头数
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        # 设置Transformer前馈网络维度
        transformer_dim_feedforward = 1024
        # 获取Transformer编码器层数
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        # 获取卷积维度
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        # 获取掩码维度
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        # 获取Transformer编码器输入特征列表
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        # 创建MSDeformAttnPixelDecoder实例
        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,  # 输入尺寸
                                                 transformer_dropout,  # dropout率
                                                 transformer_nheads,  # 注意力头数
                                                 transformer_dim_feedforward,  # tf前馈网维度
                                                 transformer_enc_layers,  # 编码层数量
                                                 conv_dim,  # 卷积层维度
                                                 mask_dim,  # 掩码维度
                                                 transformer_in_features,  # 主干网络输出特征？
                                                 common_stride)  # 步长
        # 返回像素解码器实例
        return pixel_decoder

    # 预测器初始化函数
    def predictor_init(self, cfg):
        # 获取输入通道数
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        # 获取类别数
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # 获取隐藏层维度
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        # 获取对象查询数量
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # 获取注意力头数
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        # 获取前馈网络维度
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        # 获取解码器层数（减1） ？ 为什么要减一
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        # 获取是否使用预归一化
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        # 获取掩码维度
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        # 设置是否强制输入投影
        enforce_input_project = False
        # 设置是否进行掩码分类
        mask_classification = True
        # 创建MultiScaleMaskedTransformerDecoder实例
        predictor = MultiScaleMaskedTransformerDecoder(in_channels,  # 输入通道数量
                                                       num_classes,  # 类别数量
                                                       mask_classification,  # 掩码分类
                                                       hidden_dim,  # 隐藏层维度
                                                       num_queries,  # 查询数量
                                                       nheads,  # 注意力头数
                                                       dim_feedforward,  # 前馈网络维度
                                                       dec_layers,  # 解码层个数
                                                       pre_norm,  # 是否使用归一化
                                                       mask_dim,  # 获取掩码维度
                                                       enforce_input_project)  # 是否强制输出索引
        # 返回预测器实例
        return predictor

    # 前向传播函数
    def forward(self, features, mask=None):
        # 通过像素解码器提取特征
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        # 通过预测器生成预测结果
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        # 返回预测结果
        return predictions


# 定义MaskFormer模型类，继承自nn.Module
class MaskFormerModel(nn.Module):
    # 初始化函数，接收配置参数
    def __init__(self, cfg):
        # 调用父类初始化函数
        super().__init__()
        # 构建骨干网络
        self.backbone = self.build_backbone(cfg)
        # 构建语义分割头部
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)

    # 构建骨干网络函数
    def build_backbone(self, cfg):
        # 获取模型类型
        model_type = cfg.MODEL.BACKBONE.TYPE
        model_version = cfg.MODEL.BACKBONE.VERSION
        # 如果是resnet类型
        if model_type == 'resnet':
            # 定义通道数列表
            channels = [64, 128, 256, 512]
            # 如果ResNet深度大于34，通道数需要乘以4
            if cfg.MODEL.RESNETS.DEPTH > 34:
                channels = [item * 4 for item in channels]
            # 创建ResNet实例
            backbone = ResNet(resnet_spec[model_type + model_version][0], resnet_spec[model_type + model_version][1])
            # backbone.init_weights()
            # 初始化骨干网络特征形状字典
            self.backbone_feature_shape = dict()
            # 遍历通道数，构建特征形状字典
            for i, channel in enumerate(channels):
                self.backbone_feature_shape[f'res{i + 2}'] = Dict({'channel': channel, 'stride': 2 ** (i + 2)})
        # 如果是swin类型
        elif model_type == 'swin':
            # 定义Swin Transformer的深度配置
            swin_depth = {'tiny': [2, 2, 6, 2], 'small': [2, 2, 18, 2], 'base': [2, 2, 18, 2], 'large': [2, 2, 18, 2]}
            # 定义Swin Transformer的头数配置
            swin_heads = {'tiny': [3, 6, 12, 24], 'small': [3, 6, 12, 24], 'base': [4, 8, 16, 32],
                          'large': [6, 12, 24, 48]}
            # 定义Swin Transformer的嵌入维度配置
            swin_dim = {'tiny': 96, 'small': 96, 'base': 128, 'large': 192}
            # 设置Swin Transformer的深度参数
            cfg.MODEL.SWIN.DEPTHS = swin_depth[cfg.MODEL.SWIN.TYPE]
            # 设置Swin Transformer的头数参数
            cfg.MODEL.SWIN.NUM_HEADS = swin_heads[cfg.MODEL.SWIN.TYPE]
            # 设置Swin Transformer的嵌入维度参数
            cfg.MODEL.SWIN.EMBED_DIM = swin_dim[cfg.MODEL.SWIN.TYPE]
            # 创建D2SwinTransformer实例
            backbone = D2SwinTransformer(cfg)
            # 获取骨干网络输出形状
            self.backbone_feature_shape = backbone.output_shape()
        else:
            # 如果模型类型不支持，抛出NotImplementedError异常
            raise NotImplementedError('Do not support model type!')
        # 返回骨干网络实例
        return backbone

    # 前向传播函数  inputs:[9, 3, 576, 1024]
    def forward(self, inputs):
        # 通过骨干网络提取特征
        features = self.backbone(inputs)  #
        # 通过语义分割头部生成输出
        outputs = self.sem_seg_head(features)
        # 返回输出结果
        return outputs