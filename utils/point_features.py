import torch
# 导入PyTorch的神经网络函数模块
from torch.nn import functional as F

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid. 输入是不同通道的特征图 torch.Size([4, 1, 144, 256])
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains. 输入的是随机点位
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.

    一个对 :function:torch.nn.functional.grid_sample 的封装，支持 3D point_coords 张量。
    与 :function:torch.nn.functional.grid_sample 不同的是，此函数假设 point_coords 位于
    [0, 1] x [0, 1] 的正方形范围内。

    参数:
        input (Tensor): 形状为 (N, C, H, W) 的张量，包含 H x W 网格上的特征图。输入是不同通道的特征图 torch.Size([4, 1, 144, 256])
        point_coords (Tensor): 形状为 (N, P, 2) 或 (N, Hgrid, Wgrid, 2) 的张量，包含
        [0, 1] x [0, 1] 范围内的归一化点坐标。输入的是随机点位

    返回:
        output (Tensor): 形状为 (N, C, P) 或 (N, C, Hgrid, Wgrid) 的张量，包含
            `point_coords` 中各点处的特征值。这些特征通过双线性插值从 `input` 中获取，
            其方式与 :function:`torch.nn.functional.grid_sample` 相同。

    """
    # 初始化标志位，用于标记是否需要添加维度
    add_dim = False
    # 检查点坐标的维度是否为3
    if point_coords.dim() == 3:
        # 设置标志位为True，表示需要添加维度
        add_dim = True
        # 在第2个维度上增加一个维度，形状变为 [N, P, 1, 2]
        point_coords = point_coords.unsqueeze(2) # [c, self.num_points, 1, 2] torch.Size([4, 12544, 1, 2])
    # 使用grid_sample函数进行采样，将坐标从[0,1]映射到[-1,1]范围
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs) # [c, 1, self.num_points, 1] 这里的隐藏参数是{'align_corners': False}
    # 如果之前添加了维度，则需要移除多余的维度
    if add_dim:
        # 移除第3个维度
        output = output.squeeze(3)
    # 返回采样结果 采样结果不是一个二位点位
    return output # [c, 1, self.num_points]

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.     #类别空间 或者 与类别无关的预测
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).   # 进行点的对数预测
        num_points (int): The number of points P to sample.  #采样点的的个数
        oversample_ratio (int): Oversampling parameter.  #过度采样比例
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.  #重要的采样比例

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    根据不确定性，在 [0, 1] x [1] 坐标空间中采样点。不确定性是通过一个名为 uncertainty_func 的函数为每个点计算的，该函数以点的 logit 预测作为输入。
    详见 PointRend 论文。
    参数：
    coarse_logits（张量）：形状为 (N, C, Hmask, Wmask) 或 (N, 1, Hmask, Wmask) 的张量，分别表示类别特定或类别不可知的预测。
    uncertainty_func：一个函数，接收形状为 (N, C, P) 或 (N, 1, P) 的张量（包含 P 个点的 logit 预测），并返回一个形状为 (N, 1, P) 的张量，表示这些点的不确定性。
    num_points（整数）：要采样的点数 P。
    oversample_ratio（整数）：过采样参数。
    importance_sample_ratio（浮点数）：通过重要性采样得到的点的比例。
    返回：
    point_coords（张量）：形状为 (N, P, 2) 的张量，包含采样得到的 P 个点的坐标。
    """
    # 断言检查：oversample_ratio必须大于等于1
    assert oversample_ratio >= 1
    # 断言检查：importance_sample_ratio必须在[0,1]范围内
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    # 获取粗略预测logits的第一个维度大小（批次大小）
    num_boxes = coarse_logits.shape[0]
    # 计算需要采样的点数，等于目标点数乘以过采样比率
    num_sampled = int(num_points * oversample_ratio)
    # 在[0,1]范围内随机生成坐标点，形状为[N, num_sampled, 2]
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    # 对随机生成的坐标点进行特征采样，获取对应的logits值  对于原始坐标点进行采样
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    '''
    必须基于对点位采样后的预测值来计算不确定性。
    若先在粗预测上算好不确定性，再对点位采样，会得到错误结果。
    举例说明：设 uncertainty_func(logits) = -abs(logits)，
    某采样点落在两个 logits 分别为 -1 和 1 的粗预测之间，其 logits 为 0，
    对应的不确定性为 0。
    但如果先在粗预测上算不确定性，两处都是 -1，采样点也会得到 -1 的不确定性，
    这与正确结果不符。
    '''
    # 使用不确定度函数计算每个采样点的不确定度
    point_uncertainties = uncertainty_func(point_logits)
    # 计算通过重要性采样的点数
    num_uncertain_points = int(importance_sample_ratio * num_points)
    # 计算随机采样的点数  采样数量减去 重要采样点
    num_random_points = num_points - num_uncertain_points
    # 获取不确定度最高的前k个点的索引  num_uncertain_point 进行选择   topk
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    # 创建偏移量，用于将相对索引转换为绝对索引 ？
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    # 将相对索引转换为绝对索引
    idx += shift[:, None]
    # 根据索引提取最重要的点坐标  对索引进行展平 之前的预测点也是做同样的操作。 然后选择的不确定点位，这里是通过随机点位进行选择的
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    # 如果还需要随机采样的点，则添加随机点坐标
    if num_random_points > 0:
        # 将重要性采样的点和随机采样的点连接起来 这里的是随机点是再进行随机采样的点位置
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    # 返回最终的点坐标
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    # 获取不确定度图的形状信息
    R, _, H, W = uncertainty_map.shape
    # 计算高度方向上的步长
    h_step = 1.0 / float(H)
    # 计算宽度方向上的步长
    w_step = 1.0 / float(W)

    # 确保采样点数不超过网格总点数
    num_points = min(H * W, num_points)
    # 获取不确定度最高的前num_points个点的索引
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    # 初始化点坐标的张量
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    # 计算x坐标：将一维索引转换为二维坐标并归一化
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    # 计算y坐标：将一维索引转换为二维坐标并归一化
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    # 返回点索引和点坐标
    return point_indices, point_coords