# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine


class SelfAttentionLayer(nn.Module):
    """
    自注意力层，用于处理查询特征之间的关系
    """

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        """
        初始化自注意力层
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在注意力前进行归一化
        """
        super().__init__()
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 层归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数和归一化顺序设置
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置模型参数，使用 Xavier 均匀初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码添加到张量中
        Args:
            tensor: 输入张量
            pos: 位置编码
        Returns:
            添加位置编码后的张量
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        前向传播（后归一化）
        Args:
            tgt: 目标序列
            tgt_mask: 目标掩码
            tgt_key_padding_mask: 目标键填充掩码
            query_pos: 查询位置编码
        Returns:
            处理后的目标序列
        """
        q = k = self.with_pos_embed(tgt, query_pos)  # 融合为位置的目标序列
        # 执行自注意力计算  tgt_key_padding_mask 标记序列中的填充位置，防止模型关注那些由于批处理而填充的无效位置 这里面本质是多头注意力机制
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接和归一化
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        前向传播（前归一化）
        Args:
            tgt: 目标序列
            tgt_mask: 目标掩码
            tgt_key_padding_mask: 目标键填充掩码
            query_pos: 查询位置编码
        Returns:
            处理后的目标序列
        """
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # 执行自注意力计算
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 残差连接
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        前向传播主函数
        Args:
            tgt: 目标序列
            tgt_mask: 目标掩码
            tgt_key_padding_mask: 目标键填充掩码
            query_pos: 查询位置编码
        Returns:
            处理后的目标序列
        """
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层，用于处理查询和记忆特征之间的关系
    """

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        """
        初始化交叉注意力层
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在注意力前进行归一化
        """
        super().__init__()
        # 多头交叉注意力机制
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 层归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数和归一化顺序设置
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置模型参数，使用 Xavier 均匀初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码添加到张量中
        Args:
            tensor: 输入张量
            pos: 位置编码
        Returns:
            添加位置编码后的张量
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        前向传播（后归一化）
        Args:
            tgt: 目标序列（查询）
            memory: 记忆序列（键和值）
            memory_mask: 记忆掩码
            memory_key_padding_mask: 记忆键填充掩码
            pos: 记忆位置编码
            query_pos: 查询位置编码
        Returns:
            处理后的目标序列
        """
        # 执行交叉注意力计算
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # 残差连接和归一化
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        """
        前向传播（前归一化）
        Args:
            tgt: 目标序列（查询）
            memory: 记忆序列（键和值）
            memory_mask: 记忆掩码
            memory_key_padding_mask: 记忆键填充掩码
            pos: 记忆位置编码
            query_pos: 查询位置编码
        Returns:
            处理后的目标序列
        """
        tgt2 = self.norm(tgt)
        # 执行交叉注意力计算
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # 残差连接
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        前向传播主函数
        Args:
            tgt: 目标序列（查询）
            memory: 记忆序列（键和值）
            memory_mask: 记忆掩码
            memory_key_padding_mask: 记忆键填充掩码
            pos: 记忆位置编码
            query_pos: 查询位置编码
        Returns:
            处理后的目标序列
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """
    前馈网络层（Feed-Forward Network）
    """

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        """
        初始化前馈网络层
        Args:
            d_model: 模型维度
            dim_feedforward: 前馈网络隐藏层维度
            dropout: dropout比率
            activation: 激活函数类型
            normalize_before: 是否在前馈网络前进行归一化
        """
        super().__init__()
        # 前馈网络的两层线性变换 Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

        # 激活函数和归一化顺序设置
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置模型参数，使用 Xavier 均匀初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码添加到张量中
        Args:
            tensor: 输入张量
            pos: 位置编码
        Returns:
            添加位置编码后的张量
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        """
        前向传播（后归一化）
        Args:
            tgt: 输入张量
        Returns:
            处理后的张量
        """
        # 两层线性变换和激活函数
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 残差连接和归一化
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        """
        前向传播（前归一化）
        Args:
            tgt: 输入张量
        Returns:
            处理后的张量
        """
        tgt2 = self.norm(tgt)
        # 两层线性变换和激活函数
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # 残差连接
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        """
        前向传播主函数
        Args:
            tgt: 输入张量
        Returns:
            处理后的张量
        """
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """
    根据字符串返回对应的激活函数
    Args:
        activation: 激活函数名称
    Returns:
        对应的激活函数
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron）
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        初始化多层感知机
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
        """
        super().__init__()
        self.num_layers = num_layers
        # 构建隐藏层维度列表
        h = [hidden_dim] * (num_layers - 1)
        # 创建线性层列表
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            处理后的张量
        """
        for i, layer in enumerate(self.layers):
            # 对除最后一层外的所有层应用ReLU激活函数
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    """
    多尺度掩码Transformer解码器
    """

    def __init__(
            self,
            in_channels,
            num_classes,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=10,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False
    ):
        """
        初始化多尺度掩码Transformer解码器
        Args:
            in_channels: 输入通道数
            num_classes: 类别数
            mask_classification: 是否进行掩码分类
            hidden_dim: 隐藏层维度
            num_queries: 查询数量
            nheads: 注意力头数
            dim_feedforward: 前馈网络维度
            dec_layers: 解码器层数
            pre_norm: 是否使用前置归一化
            mask_dim: 掩码维度
            enforce_input_project: 是否强制输入投影
        """
        super().__init__()

        # 断言必须支持掩码分类
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # 位置编码 positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # 定义Transformer解码器
        self.num_heads = nheads
        self.num_layers = dec_layers
        # 创建自注意力、交叉注意力和前馈网络层列表
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        # 创建指定数量的各类型层
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        # 解码器归一化层 hidden_dim 指 “最后一个维度的大小”
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # 可学习的查询特征和位置编码
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 层级嵌入（始终使用3个尺度） level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        # 为每个特征层级创建输入投影层
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # 输出前馈网络
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        """
        前向传播
        Args:
            x: 多尺度特征列表
            mask_features: 掩码特征
            mask: 掩码（不使用）
        Returns:
            解码结果字典
        """
        # x 是多尺度特征列表
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []  # 保存每个特征图下面的尺寸

        # 禁用掩码，因为它不影响性能  disable mask, it does not affect performance
        del mask

        # 处理每个特征层级
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # 生成位置编码并展平 这里一样使用的是三角函数位置编码
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # 输入投影并添加层级嵌入 对每一层使用的投影卷积都是不一样的 将图像展平之后与层级嵌入进行广播相加
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # 将 NxCxHxW 转换为 HWxNxC 也就是将当前为欸都进行提取，然后进行转化维度 将图片特征维度提取到最前
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape  # 这一步只是为了获取batch_size

        # QxNxC 查询嵌入和特征 添加batch_size 维度  todo 为什么是在中间进行添加 因为在多头注意力机制中需要这样的输入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # 在可学习查询特征上进行预测头计算 prediction heads on learnable query features 第一次是初始化
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)  # 类别预测
        predictions_mask.append(outputs_mask)

        # 遍历每个解码器层
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # 处理注意力掩码 对全部都是掩码的部分赋值为False 防止出现全屏蔽情况，这样做的目的是为了防止出现数值问题
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # 注意：先进行交叉注意力  attention: cross-attention first 这里对第i层次的 attention 进行处理
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # 这里不对填充区域应用掩码  here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed
            )

            # 自注意力
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # 前馈网络  FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            # 预测头计算  这里是对每层的结果进行计算
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                   attn_mask_target_size=size_list[(
                                                                                                                               i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # 确保预测数量正确
        assert len(predictions_class) == self.num_layers + 1

        # 构建输出字典
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(  # 辅助损失 只是将内容保存起来
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """
        前向传播预测头
        Args:
            output: 解码器输出
            mask_features: 掩码特征
            attn_mask_target_size: 注意力掩码目标尺寸
        Returns:
            分类输出、掩码输出和注意力掩码
        """
        # 解码器归一化和转置
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # 分类预测 nn.Linear(hidden_dim, num_classes + 1)
        outputs_class = self.class_embed(decoder_output)
        # 掩码嵌入 n个线性层
        mask_embed = self.mask_embed(decoder_output)
        # 生成掩码输出 爱因斯坦求和约定 这里是张量收缩？ 将mask_features中的每个像素位置上的嵌入层，和mask_embed的每个嵌入层进行计算
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # 注意：预测结果具有更高分辨率   NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        # 插值到目标尺寸
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # 必须使用布尔类型
        # 如果提供BoolTensor，则不允许值为True的位置参与注意力计算，False值保持不变
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.

        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()  # 为什么这里要脱离计算图

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """
        设置辅助损失（这是一个解决torchscript问题的变通方法）
        Args:
            outputs_class: 分类输出
            outputs_seg_masks: 分割掩码输出
        Returns:
            辅助输出列表
        """
        # 这是一个解决torchscript问题的变通方法，因为torchscript
        # 不支持具有非同类值的字典，例如包含张量和列表的字典
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]