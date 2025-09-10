# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import sys
import os

# 将上级目录添加到系统路径中，以便导入自定义模块
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

# 导入点特征相关函数和辅助函数
from .point_features import point_sample, get_uncertain_point_coords_with_randomness
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, get_world_size


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # 对输入进行sigmoid激活，将其转换为概率值
    inputs = inputs.sigmoid()
    # 将输入展平为二维张量，保留批次维度
    inputs = inputs.flatten(1)
    # 计算预测值和目标值的交集
    numerator = 2 * (inputs * targets).sum(-1)
    # 计算预测值和目标值的并集
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算Dice损失，添加1是为了避免除零错误
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对所有样本的损失求和并除以mask数量进行归一化
    return loss.sum() / num_masks


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # 使用sigmoid二元交叉熵计算损失，不进行归约
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 对每个样本计算均值，然后求和并除以mask数量进行归一化
    return loss.mean(1).sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # 计算sigmoid概率值
    prob = inputs.sigmoid()
    # 计算二元交叉熵损失
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 计算p_t值，用于Focal Loss的调制因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 应用Focal Loss调制因子
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果提供了alpha参数，则应用类别权重平衡
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 对每个样本计算均值，然后求和并除以mask数量进行归一化
    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    在语义分割中，logits是模型输出的原始分数（未经过sigmoid或softmax），这些分数反映了模型对每个像素属于某个类别的置信度：
    高正值（如+10）：模型非常确定该像素属于前景
    高负值（如-10）：模型非常确定该像素属于背景
    接近0的值（如0.1, -0.2）：模型不太确定该像素的类别
    """
    # 确保logits只有一个通道
    assert logits.shape[1] == 1
    # 克隆logits张量
    gt_class_logits = logits.clone()
    # 计算不确定性分数，使用L1距离 绝对值公式  这里添加符号后，值越大那么就越是不确定的值
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, device):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        # 调用父类初始化方法
        super().__init__()
        # 设置类别数量
        self.num_classes = num_classes
        # 设置匹配器
        self.matcher = matcher
        # 设置损失权重字典
        self.weight_dict = weight_dict
        # 设置无对象类别的系数
        self.eos_coef = eos_coef
        # 设置需要计算的损失列表
        self.losses = losses
        # 设置设备
        self.device = device
        # 创建空类别权重张量
        empty_weight = torch.ones(self.num_classes + 1).to(device)
        # 设置最后一个类别的权重（无对象类别）
        empty_weight[-1] = self.eos_coef
        # 注册buffer，使其在模型保存/加载时被包含
        self.register_buffer("empty_weight", empty_weight)

        # 点级mask损失参数
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # 确保输出包含预测logits
        assert "pred_logits" in outputs
        # 获取预测logits并转换为float类型
        src_logits = outputs["pred_logits"].float()

        # 获取源索引
        idx = self._get_src_permutation_idx(indices)
        # 根据匹配索引获取目标类别标签 获得真是是的类目
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(self.device)
        # 创建目标类别张量，初始值为0
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        # 根据匹配索引设置目标类别
        target_classes[idx] = target_classes_o

        # 计算交叉熵损失
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # 构造损失字典 返回关于类别的损失
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # 确保输出包含预测mask
        assert "pred_masks" in outputs

        # 获取源索引和目标索引
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # 获取预测mask
        src_masks = outputs["pred_masks"]  #
        # 根据源索引选择对应的预测mask
        src_masks = src_masks[src_idx]
        # 提取目标mask
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        # 从tensor列表创建嵌套tensor并分解
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # 将目标mask移到与源mask相同的设备上
        target_masks = target_masks.to(src_masks)
        # 根据目标索引选择对应的目标mask
        target_masks = target_masks[tgt_idx]

        # ===================================================================================
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W  随机预选不确定点的做法正高明 能够十分有效的降低损失的同时性能也没有太大的下降 发明这个方法的人真的是天才 太好用了

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords  点坐标 对不确定性高的地方多采样
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels   对gt 的masks 进行采样 采样的点位是上面的point_coords
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
        # 对预测的点位 进行采样 采样的点位是上面的 point_coords
        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        # ===================================================================================
        # 将预测mask和目标mask展平
        # point_logits = src_masks.flatten(1)
        # point_labels = target_masks.flatten(1)

        # 计算mask损失和dice损失
        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            # sigmoid_focal_loss(point_logits, point_labels, num_masks), #
            "loss_dice": dice_loss(point_logits, point_labels, num_masks)
        }

        # 删除临时变量释放内存
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # 根据索引排列预测值 permute predictions following indices  填充预测原图对应的索引位置
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])  # 创建原图索引为一排
        return batch_idx, src_idx  # 为什么这里返回的是一个元组类型

    def _get_tgt_permutation_idx(self, indices):
        # 根据索引排列目标值 permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_binary_mask(self, target):
        # 将目标mask转换为二值mask
        y, x = target.size()
        # 创建one-hot编码的target 对每一个类别都设置一个掩膜
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        # 根据target值scatter设置为1  真实类被代表索引的类  eg   index 25（第25个类别） = 1（非25这个类别那么就是0）
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)  #
        return target_onehot  #

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        # 定义损失映射字典
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        # 确保请求的损失类型存在
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # 调用对应的损失计算函数
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        # 移除辅助输出，只保留主要输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # 获取目标 list(batch)  然后里面是字典 分别保存有labels以及masks
        targets = self._get_targets(gt_masks)
        # 使用匹配器获取输出和目标之间的匹配索引 Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上目标框的平均数量，用于归一化 Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        # 如果是分布式训练，则进行all_reduce操作
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        # 限制最小值为1并转换为标量 todo 为什么要限制大小
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # 计算所有请求的损失 Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # 如果存在辅助输出，则对每个辅助输出重复上述过程 In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_targets(self, gt_masks):
        # 根据ground truth masks生成目标格式
        targets = []
        for mask in gt_masks:  # 获取批次里面 每一个值
            # 将mask转换为二值mask
            binary_masks = self._get_binary_mask(mask)
            # 获取唯一的类别标签
            cls_label = torch.unique(mask)
            # 排除背景类（标签0）
            labels = cls_label[1:]
            # 选择对应类别的二值mask
            binary_masks = binary_masks[labels]
            # 添加到目标列表中
            targets.append({'masks': binary_masks, 'labels': labels})
        return targets

    def __repr__(self):
        # 重写类的字符串表示方法
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class Criterion(object):
    def __init__(self, num_classes, alpha=0.5, gamma=2, weight=None, ignore_index=0):
        # 初始化Criterion类
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = 1e-5
        # 定义交叉熵损失函数
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')

    def get_loss(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        # 初始化各类损失
        loss_labels = 0.0
        loss_masks = 0.0
        loss_dices = 0.0
        # 获取批次大小
        num = gt_masks.shape[0]
        # 获取预测logits和mask
        pred_logits = [outputs["pred_logits"].float()]  # [bs, num_query, num_classes + 1]
        pred_masks = [outputs['pred_masks'].float()]  # [bs, num_query, h, w]
        # 获取目标
        targets = self._get_targets(gt_masks, pred_logits[0].shape[1], pred_logits[0].device)
        # 处理辅助输出
        for aux_output in outputs['aux_outputs']:
            pred_logits.append(aux_output["pred_logits"].float())
            pred_masks.append(aux_output["pred_masks"].float())

        # 获取目标标签和mask列表
        gt_label = targets['labels']  # [bs, num_query]
        gt_mask_list = targets['masks']
        # 遍历预测logits和mask计算损失
        for mask_cls, pred_mask in zip(pred_logits, pred_masks):
            # 计算标签损失
            loss_labels += F.cross_entropy(mask_cls.transpose(1, 2), gt_label)
            # loss_masks += self.focal_loss(pred_result, gt_masks.to(pred_result.device))
            # 计算dice损失
            loss_dices += self.dice_loss(pred_mask, gt_mask_list)

        # 返回平均损失
        return loss_labels / num, loss_dices / num

    def binary_dice_loss(self, inputs, targets):
        # 计算二值dice损失
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        # 计算分子（交集）
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        # 计算分母（并集）
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        # 计算dice损失
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def dice_loss(self, predict, targets):
        # 计算dice损失
        bs = predict.shape[0]
        total_loss = 0
        # 遍历每个样本
        for i in range(bs):
            pred_mask = predict[i]
            tgt_mask = targets[i].to(predict.device)
            # 计算单个样本的dice损失
            dice_loss_value = self.binary_dice_loss(pred_mask, tgt_mask)
            total_loss += dice_loss_value
        # 返回平均损失
        return total_loss / bs

    def focal_loss(self, preds, labels):
        """
        preds: [bs, num_class + 1, h, w]
        labels: [bs, h, w]
        """
        # 计算focal loss
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss.mean()

    def _get_binary_mask(self, target):
        # 将目标转换为二值mask
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def _get_targets(self, gt_masks, num_query, device):
        # 初始化二值mask列表
        binary_masks = []
        # 初始化标签列表
        gt_labels = []
        # 遍历每一张ground truth mask
        for mask in gt_masks:
            # 将mask转换为one-hot编码
            mask_onehot = self._get_binary_mask(mask)
            # 获取mask中唯一的类别标签
            cls_label = torch.unique(mask)
            # 创建与预测结果相同长度的标签张量，初始值为0
            labels = torch.full((num_query,), 0, dtype=torch.int64, device=gt_masks.device)
            # 将真实的类别标签填入对应位置
            labels[:len(cls_label)] = cls_label
            # 提取对应类别的二值mask并添加到列表中
            binary_masks.append(mask_onehot[cls_label])
            # 将类别标签添加到列表中
            gt_labels.append(labels)
        # 返回处理后的标签和mask
        return {"labels": torch.stack(gt_labels).to(device), "masks": binary_masks}