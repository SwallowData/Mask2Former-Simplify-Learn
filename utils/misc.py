# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
# 从typing模块导入List和Optional类型提示
from typing import List, Optional
# 从collections模块导入OrderedDict有序字典
from collections import OrderedDict
# 从scipy.io导入loadmat函数，用于加载MATLAB文件
from scipy.io import loadmat
# 导入numpy库，用于数值计算
import numpy as np
# 导入csv模块，用于处理CSV文件
import csv
# 从PIL库导入Image模块，用于图像处理
from PIL import Image
# 从matplotlib.pyplot导入pyplot接口，用于绘图
import matplotlib.pyplot as plt
# 导入torch库，PyTorch深度学习框架的核心
import torch
# 导入torch.distributed模块，用于分布式训练
import torch.distributed as dist
# 导入torchvision库，用于计算机视觉任务
import torchvision
# 从torch模块导入Tensor类型
from torch import Tensor


# 定义一个函数，用于获取多维列表中每个维度的最大值
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    # 初始化最大值列表为第一个子列表
    maxes = the_list[0]
    # 遍历剩余的子列表
    for sublist in the_list[1:]:
        # 遍历子列表中的每个元素及其索引
        for index, item in enumerate(sublist):
            # 更新对应位置的最大值
            maxes[index] = max(maxes[index], item)
    # 返回每个维度的最大值列表
    return maxes

# 获取当前分布式环境中的进程数量
def get_world_size() -> int:
    # 检查分布式训练是否可用
    if not dist.is_available():
        # 如果不可用，返回1（单机模式）
        return 1
    # 检查分布式训练是否已初始化
    if not dist.is_initialized():
        # 如果未初始化，返回1（单机模式）
        return 1
    # 返回当前分布式环境中的进程总数
    return dist.get_world_size()

# 对字典中的值进行分布式规约操作
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    # 获取当前分布式环境中的进程数量
    world_size = get_world_size()
    # 如果进程数小于2，则无需规约，直接返回原字典
    if world_size < 2:
        return input_dict
    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 初始化键名和值的列表
        names = []
        values = []
        # 对字典键进行排序，确保各进程键的顺序一致
        for k in sorted(input_dict.keys()):
            # 将键名添加到names列表
            names.append(k)
            # 将对应的值添加到values列表
            values.append(input_dict[k])
        # 将值列表堆叠成张量
        values = torch.stack(values, dim=0)
        # 对所有进程的张量进行规约操作（求和）
        dist.all_reduce(values)
        # 如果需要计算平均值
        if average:
            # 将规约后的值除以进程数得到平均值
            values /= world_size
        # 构建规约后的字典
        reduced_dict = {k: v for k, v in zip(names, values)}
    # 返回规约后的字典
    return reduced_dict

# 定义NestedTensor类，用于处理不同尺寸的张量
class NestedTensor(object):
    # 初始化函数，接受张量和掩码作为参数
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 保存张量数据
        self.tensors = tensors
        # 保存掩码数据
        self.mask = mask

    # 定义to方法，将张量移动到指定设备
    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        # 将张量移动到指定设备
        cast_tensor = self.tensors.to(device)
        # 获取掩码
        mask = self.mask
        # 如果掩码不为空
        if mask is not None:
            # 确保掩码不为空
            assert mask is not None
            # 将掩码移动到指定设备
            cast_mask = mask.to(device)
        else:
            # 如果掩码为空，则设置为None
            cast_mask = None
        # 返回新的NestedTensor对象
        return NestedTensor(cast_tensor, cast_mask)

    # 定义decompose方法，分解张量和掩码
    def decompose(self):
        # 返回张量和掩码
        return self.tensors, self.mask

    # 定义__repr__方法，用于打印对象信息
    def __repr__(self):
        # 返回张量的字符串表示
        return str(self.tensors)

# 从张量列表创建NestedTensor对象
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # 检查第一个张量的维度是否为3
    if tensor_list[0].ndim == 3:
        # 检查是否在进行ONNX追踪
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            # 如果在进行ONNX追踪，调用ONNX兼容的版本
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        # 计算所有张量的最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        # 计算批处理的形状，包括批次维度
        batch_shape = [len(tensor_list)] + max_size  #加上batch
        # 解包批次形状为b,c,h,w
        b, c, h, w = batch_shape
        # 获取第一个张量的数据类型
        dtype = tensor_list[0].dtype
        # 获取第一个张量的设备
        device = tensor_list[0].device
        # 创建一个全零张量作为批处理张量
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 创建一个全1的布尔型掩码张量
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 遍历张量列表、批处理张量和掩码
        for img, pad_img, m in zip(tensor_list, tensor, mask): #pad_img 和 m 都是随机初始化的值
            # 将原始图像复制到填充图像中
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # 更新掩码，将有效区域设为False
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果张量维度不是3，抛出异常
        raise ValueError("not supported")
    # 返回NestedTensor对象
    return NestedTensor(tensor, mask)

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
# 定义ONNX兼容版本的nested_tensor_from_tensor_list函数
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    # 初始化最大尺寸列表
    max_size = []
    # 遍历第一个张量的每个维度
    for i in range(tensor_list[0].dim()):
        # 计算该维度上的最大尺寸
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        # 将最大尺寸添加到列表中
        max_size.append(max_size_i)
    # 将最大尺寸列表转换为元组
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    # 初始化填充图像和掩码列表
    padded_imgs = []
    padded_masks = []
    # 遍历张量列表中的每个图像
    for img in tensor_list:
        # 计算每个维度需要填充的大小
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        # 对图像进行填充
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        # 将填充后的图像添加到列表中
        padded_imgs.append(padded_img)

        # 创建掩码张量
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        # 对掩码进行填充
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        # 将填充后的掩码添加到列表中，并转换为布尔类型
        padded_masks.append(padded_mask.to(torch.bool))

    # 将填充后的图像堆叠成张量
    tensor = torch.stack(padded_imgs)
    # 将填充后的掩码堆叠成张量
    mask = torch.stack(padded_masks)

    # 返回NestedTensor对象
    return NestedTensor(tensor, mask=mask)

# 检查分布式训练是否可用且已初始化
def is_dist_avail_and_initialized():
    # 检查分布式训练是否可用
    if not dist.is_available():
        # 如果不可用，返回False
        return False
    # 检查分布式训练是否已初始化
    if not dist.is_initialized():
        # 如果未初始化，返回False
        return False
    # 如果都满足，返回True
    return True

# 加载并行模型的权重
def load_parallal_model(model, state_dict_):
    # 创建有序字典用于存储处理后的状态字典
    state_dict = OrderedDict()
    # 遍历原始状态字典的键
    for key in state_dict_:
        # 如果键以'module'开头但不以'module_list'开头
        if key.startswith('module') and not key.startswith('module_list'):
            # 去除'module.'前缀后添加到新字典中
            state_dict[key[7:]] = state_dict_[key]
        else:
            # 否则直接添加到新字典中
            state_dict[key] = state_dict_[key]

    # check loaded parameters and created model parameters
    # 获取模型的当前状态字典
    model_state_dict = model.state_dict()
    # 遍历处理后的状态字典
    for key in state_dict:
        # 如果键在模型状态字典中存在
        if key in model_state_dict:
            # 检查形状是否匹配
            if state_dict[key].shape != model_state_dict[key].shape:
                # 如果形状不匹配，打印警告信息并使用模型的参数
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            # 如果键不存在，打印提示信息
            print('Drop parameter {}.'.format(key))
    # 遍历模型状态字典
    for key in model_state_dict:
        # 如果键不在处理后的状态字典中
        if key not in state_dict:
            # 打印提示信息并添加模型的参数
            print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    # 加载处理后的状态字典到模型中
    model.load_state_dict(state_dict, strict=False)

    # 返回模型
    return model

# ADE20K数据集可视化类
class ADEVisualize(object):
    # 初始化函数
    def __init__(self):
        # 加载颜色映射文件
        self.colors = loadmat('dataset/color150.mat')['colors']
        # 初始化名称字典
        self.names = {}
        # 打开对象信息CSV文件
        with open('dataset/object150_info.csv') as f:
            # 创建CSV读取器
            reader = csv.reader(f)
            # 跳过表头行
            next(reader)
            # 遍历每一行数据
            for row in reader:
                # 将类别ID和名称存储到字典中
                self.names[int(row[0])] = row[5].split(";")[0]

    # 获取数组中的唯一值
    def unique(self, ar, return_index=False, return_inverse=False, return_counts=False):
        # 将输入数组展平
        ar = np.asanyarray(ar).flatten()

        # 判断是否需要返回索引或逆序索引
        optional_indices = return_index or return_inverse
        # 判断是否需要返回结果
        optional_returns = optional_indices or return_counts

        # 如果数组为空
        if ar.size == 0:
            # 如果不需要返回结果
            if not optional_returns:
                ret = ar
            else:
                # 构建返回元组
                ret = (ar,)
                # 如果需要返回索引
                if return_index:
                    ret += (np.empty(0, np.bool),)
                # 如果需要返回逆序索引
                if return_inverse:
                    ret += (np.empty(0, np.bool),)
                # 如果需要返回计数
                if return_counts:
                    ret += (np.empty(0, np.intp),)
            # 返回结果
            return ret
        # 如果需要返回索引
        if optional_indices:
            # 对数组进行排序，获取排序索引
            perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
            # 根据排序索引重新排列数组
            aux = ar[perm]
        else:
            # 对数组进行排序
            ar.sort()
            # 保存排序后的数组
            aux = ar
        # 判断相邻元素是否相等，生成标志数组
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))

        # 如果不需要返回结果
        if not optional_returns:
            ret = aux[flag]
        else:
            # 构建返回元组
            ret = (aux[flag],)
            # 如果需要返回索引
            if return_index:
                ret += (perm[flag],)
            # 如果需要返回逆序索引
            if return_inverse:
                # 计算累积和并减1
                iflag = np.cumsum(flag) - 1
                # 创建空的逆序索引数组
                inv_idx = np.empty(ar.shape, dtype=np.intp)
                # 根据排序索引填充逆序索引数组
                inv_idx[perm] = iflag
                ret += (inv_idx,)
            # 如果需要返回计数
            if return_counts:
                # 计算非零元素的索引并连接数组大小
                idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
                # 计算索引差值
                ret += (np.diff(idx),)
        # 返回结果
        return ret

    # 对标签图进行颜色编码
    def colorEncode(self, labelmap, colors, mode='RGB'):
        # 将标签图转换为整数类型
        labelmap = labelmap.astype('int')
        # 创建RGB格式的标签图
        labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                                dtype=np.uint8)
        # 遍历标签图中的唯一标签
        for label in self.unique(labelmap):
            # 如果标签小于0，跳过
            if label < 0:
                continue
            # 根据标签值为对应区域着色
            labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                np.tile(colors[label],
                        (labelmap.shape[0], labelmap.shape[1], 1))

        # 根据模式返回相应的颜色编码图像
        if mode == 'BGR':
            return labelmap_rgb[:, :, ::-1]
        else:
            return labelmap_rgb

    # 显示分割结果
    def show_result(self, img, pred, save_path=None):
        # 将预测结果转换为32位整数
        pred = np.int32(pred)
        # colorize prediction
        # 对预测结果进行颜色编码
        pred_color = self.colorEncode(pred, self.colors)
        # 将输入图像转换为RGBA模式
        pil_img = img.convert('RGBA')
        # 将预测颜色图转换为RGBA模式
        pred_color = Image.fromarray(pred_color).convert('RGBA')
        # 将原始图像和预测颜色图进行混合
        im_vis = Image.blend(pil_img, pred_color, 0.6)
        # 如果指定了保存路径
        if save_path is not None:
            # 保存混合后的图像
            im_vis.save(save_path)
            # Image.fromarray(im_vis).save(save_path)
        else:
            # 否则显示图像
            plt.imshow(im_vis)