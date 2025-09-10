# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from .point_features import point_sample
from torch import amp


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # 对输入进行sigmoid激活，将logits转换为概率值
    inputs = inputs.sigmoid()
    # 将输入展平为二维张量，保留批次维度，合并空间维度
    inputs = inputs.flatten(1)
    # 计算分子部分：2 * 交集，使用爱因斯坦求和约定计算预测和目标之间的交集  50个预测框对于2个真实框的交集
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    # 计算分母部分：各自元素和的和，用于Dice系数计算
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    # 计算Dice损失：1 - Dice系数，添加平滑项防止除零 类似于 generalized IOU
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    # 获取空间维度大小（高度*宽度）
    hw = inputs.shape[1]

    # 计算正样本的二元交叉熵损失（目标为1的部分）
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    # 计算负样本的二元交叉熵损失（目标为0的部分）
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    # 分别计算正负样本损失，并使用爱因斯坦求和约定进行聚合
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets)
                                                                  )

    # 对空间维度进行平均，返回平均损失
    return loss / hw


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
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
    # 获取空间维度大小
    hw = inputs.shape[1]
    # 对于分类损失 值越大代表越不确定，此时的prob代表的是一种负类的概率
    # 对输入进行sigmoid激活得到预测概率  每一个通道代表对一个类别的每一个点位的二分类预测
    prob = inputs.sigmoid()
    # 计算正样本的focal loss：((1 - prob) ** gamma) 为调制因子，关注难分样本 概率越高调解因子发挥的作用越大 对于越确信的样本权重就越小，在损失中发挥的作用就越小
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    # 计算负样本的focal loss：(prob ** gamma) 为调制因子  关注易分类样本
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    # 如果设置了alpha参数，则应用类别权重平衡 Focal Loss 通过这种机制，自动降低易分类样本的贡献，使模型更加专注于学习难分类的样本，从而提高了在类别不平衡情况下的性能。
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    # 聚合正负样本损失 正负类样本相加
    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))

    # 对空间维度进行平均
    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        # 调用父类初始化方法
        super().__init__()
        # 保存各类损失的权重系数
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        # 确保至少有一种损失权重不为0
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        # 保存采样点数量参数
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        # 获取批次大小和查询数量
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 存储匹配结果的列表
        indices = []

        # 遍历每个批次样本 Iterate through batch size
        for b in range(bs):
            # 对预测logits进行softmax得到分类概率分布
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes+1]
            # 获取预测mask
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            # 获取目标标签
            tgt_ids = targets[b]["labels"]  # [1,2,3, ……]
            # 获取目标mask并转换到与预测mask相同的设备
            tgt_mask = targets[b]["masks"].to(out_mask)  # [c, h, w] c = len(tgt_ids)

            # 计算分类成本：使用1 - 预测目标类别的概率作为成本
            # 这里省略了常数1，因为它不影响匹配结果
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]  # [num_queries, num_total_targets]

            # ===========================Mask2Former方式====================================#
            out_mask = out_mask[:, None]  # [num_queries, 1, H_pred, W_pred]
            tgt_mask = tgt_mask[:, None]  # [c, 1, h, w]

            # all masks share the same set of points for efficient matching! 随机生成像素点原本的像素点是36,864  这里采用的是从0~1之间的小数点
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,  # [c, 1, h, w]
                point_coords.repeat(tgt_mask.shape[0], 1, 1),  # [c, self.num_points, 2]
                align_corners=False,
            ).squeeze(1)  # [c, self.num_points] 通道数量 点位数量

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)  # [num_queries, self.num_points] 预测框数量 点位数量
            # ===========================end====================================#

            # ===========================MaskFormer方式====================================#
            # 展平空间维度：将mask从[H,W]展平为[H*W] Flatten spatial dimension
            # out_mask = out_mask.flatten(1)  # [num_queries, H*W]
            # tgt_mask = tgt_mask.flatten(1)  # [num_total_targets, H*W]

            # 关闭自动混合精度以确保计算精度
            with amp.autocast("cuda", enabled=False):
                # 转换为float类型以进行数值计算
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # 计算mask之间的focal loss作为匹配成本 Compute the focal loss between masks
                cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

                # 计算mask之间的dice loss作为匹配成本 Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # 计算最终的成本矩阵：加权组合各类成本 Final cost matrix
            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )
            # 重塑成本矩阵并移至CPU以进行匈牙利算法计算
            C = C.reshape(num_queries, -1).cpu()  # [num_queries, num_total_targets]

            # 使用scipy的linear_sum_assignment进行二分图匹配（匈牙利算法） 使用底层c实现
            indices.append(linear_sum_assignment(C))

        # 将匹配结果转换为tensor并返回
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # 调用memory_efficient_forward方法执行匹配
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        # 生成对象的字符串表示
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)