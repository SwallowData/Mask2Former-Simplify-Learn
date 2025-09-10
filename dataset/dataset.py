#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2023/04/06 22:39:31
@Author  :   BQH
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# 导入所需的库和模块
import os  # 操作系统接口模块，用于文件路径操作
import json  # JSON数据处理模块
import torch  # PyTorch深度学习框架
import cv2
import numpy as np  # 数值计算库
import random  # 随机数生成模块
from PIL import Image  # 图像处理模块
from PIL import ImageOps  # 图像操作模块

from copy import deepcopy  # 深拷贝模块

# 导入数据增强策略模块
from .aug_strategy import imgaug_mask
from .aug_strategy import pipe_sequential_rotate
from .aug_strategy import pipe_sequential_translate
from .aug_strategy import pipe_sequential_scale
from .aug_strategy import pipe_someof_flip
from .aug_strategy import pipe_someof_blur
from .aug_strategy import pipe_sometimes_mpshear
from .aug_strategy import pipe_someone_contrast

# 导入NuImages数据集处理模块
from .NuImages.nuimages import NuImages


def imresize(im, size, interp='bilinear'):
    # 图像缩放函数，根据指定的插值方法调整图像大小
    if interp == 'nearest':  # 最近邻插值
        resample = Image.NEAREST
    elif interp == 'bilinear':  # 双线性插值
        resample = Image.BILINEAR
    elif interp == 'bicubic':  # 双三次插值
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')  # 抛出异常，插值方法未定义

    return im.resize(size, resample)  # 返回调整大小后的图像


class BaseDataset(torch.utils.data.Dataset):
    # 基础数据集类，继承自PyTorch的Dataset类
    def __init__(self, odgt, opt, **kwargs):
        # 初始化函数，设置数据集的基本参数
        # parse options
        self.imgSizes = opt.INPUT.CROP.SIZE  # 获取图像裁剪尺寸
        self.imgMaxSize = opt.INPUT.CROP.MAX_SIZE  # 获取图像最大尺寸
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 2 ** 5  # resnet 总共下采样5次  # padding常数，用于保持网络下采样后的尺寸整除性

        # parse the input list
        if odgt is not None:  # 如果提供了odgt文件路径
            self.parse_input_list(odgt, **kwargs)  # 解析输入列表
        self.pixel_mean = np.array(opt.DATASETS.PIXEL_MEAN)  # 像素均值，用于图像标准化
        self.pixel_std = np.array(opt.DATASETS.PIXEL_STD)  # 像素标准差，用于图像标准化

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        # 解析输入列表函数，处理数据集文件列表
        if isinstance(odgt, list):  # 如果odgt是列表类型
            self.list_sample = odgt  # 直接赋值给样本列表
        elif isinstance(odgt, str):  # 如果odgt是字符串类型（文件路径）
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]  # 从文件中读取并解析JSON数据

        if max_sample > 0:  # 如果设置了最大样本数
            self.list_sample = self.list_sample[0:max_sample]  # 截取指定数量的样本
        if start_idx >= 0 and end_idx >= 0:  # 如果设置了起始和结束索引
            self.list_sample = self.list_sample[start_idx:end_idx]  # 截取指定范围的样本

        self.num_sample = len(self.list_sample)  # 获取样本总数
        assert self.num_sample > 0  # 断言样本数大于0
        print('# samples: {}'.format(self.num_sample))  # 打印样本数量

    def img_transform(self, img):
        # 图像变换函数，对图像进行标准化处理
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.  # 将图像像素值从0-255范围转换到0-1范围
        img = (img - self.pixel_mean) / self.pixel_std  # 使用均值和标准差进行标准化
        img = img.transpose((2, 0, 1))  # [c, h, w]  # 转换图像维度顺序为[channel, height, width]
        return img

    def segm_transform(self, segm: np.ndarray):
        # 分割标签变换函数，将分割标签转换为张量
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long()  # 将numpy数组转换为PyTorch张量
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        # 将x向上取整到最接近p的倍数
        return ((x - 1) // p + 1) * p

    def get_img_ratio(self, img_size, target_size):
        # 计算图像缩放比例
        img_rate = np.max(img_size) / np.min(img_size)  # 计算原始图像的长宽比
        target_rate = np.max(target_size) / np.min(target_size)  # 计算目标尺寸的长宽比
        if img_rate > target_rate:  # 如果原始图像比例大于目标比例
            # 按长边缩放
            ratio = max(target_size) / max(img_size)  # 按长边计算缩放比例
        else:
            ratio = min(target_size) / min(img_size)  # 按短边计算缩放比例
        return ratio

    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        # 图像缩放并填充函数，保持图像比例的同时进行填充
        w, h = img.size  # 获取原始图像的宽高
        target_w, target_h = outsize[0], outsize[1]  # 获取目标尺寸
        ratio = self.get_img_ratio([w, h], outsize)  # 计算缩放比例
        ow, oh = round(w * ratio), round(h * ratio)  # 计算缩放后的尺寸
        img = img.resize((ow, oh), Interpolation)  # 按比例缩放图像
        dh, dw = target_h - oh, target_w - ow  # 计算需要填充的高度和宽度
        top, bottom = dh // 2, dh - (dh // 2)  # 计算上下填充量
        left, right = dw // 2, dw - (dw // 2)  # 计算左右填充量
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针进行填充
        return img


class ADE200kDataset(BaseDataset):
    # ADE20K数据集类，继承自BaseDataset
    def __init__(self, odgt, opt, dynamic_batchHW=False, **kwargs):
        super(ADE200kDataset, self).__init__(odgt, opt, **kwargs)  # 调用父类初始化
        self.root_dataset = opt.DATASETS.ROOT_DIR  # 数据集根目录
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.MODEL.SEM_SEG_HEAD.COMMON_STRIDE  # 网络输出相对于输入缩小的倍数
        self.dynamic_batchHW = dynamic_batchHW  # 是否动态调整batchHW, cswin_transformer需要使用固定image size
        self.num_querys = opt.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES  # MaskFormer查询数量
        # self.visualize = ADEVisualize()

        self.aug_pipe = self.get_data_aug_pipe()  # 获取数据增强管道

    def get_data_aug_pipe(self):
        # 获取数据增强管道函数
        pipe_aug = []  # 初始化增强管道列表
        if random.random() > 0.5:  # 50%概率进行数据增强
            aug_list = [pipe_sequential_rotate, pipe_sequential_scale, pipe_sequential_translate, pipe_someof_blur,
                        pipe_someof_flip, pipe_sometimes_mpshear, pipe_someone_contrast]  # 增强方法列表
            index = np.random.choice(a=[0, 1, 2, 3, 4, 5, 6],
                                     p=[0.05, 0.25, 0.20, 0.25, 0.15, 0.05, 0.05])  # 随机选择增强方法
            if (index == 0 or index == 4 or index == 5) and random.random() < 0.5:  # 特定增强方法的组合策略
                index2 = np.random.choice(a=[1, 2, 3], p=[0.4, 0.3, 0.3])  # 随机选择第二个增强方法
                pipe_aug = [aug_list[index], aug_list[index2]]  # 组合两个增强方法
            else:
                pipe_aug = [aug_list[index]]  # 单个增强方法
        return pipe_aug

    def get_batch_size(self, batch_records):
        # 获取批次尺寸函数
        batch_width, batch_height = self.imgMaxSize[0], self.imgMaxSize[1]  # 初始化批次宽高

        if self.dynamic_batchHW:  # 如果启用动态批次尺寸
            if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):  # 如果图像尺寸是列表或元组
                this_short_size = np.random.choice(self.imgSizes)  # 随机选择一个尺寸
            else:
                this_short_size = self.imgSizes  # 使用固定尺寸

            batch_widths = np.zeros(len(batch_records), np.int32)  # 初始化批次宽度数组
            batch_heights = np.zeros(len(batch_records), np.int32)  # 初始化批次高度数组
            for i, item in enumerate(batch_records):  # 遍历批次记录
                img_height, img_width = item['image'].shape[0], item['image'].shape[1]  # 获取图像尺寸
                this_scale = min(
                    this_short_size / min(img_height, img_width), \
                    self.imgMaxSize / max(img_height, img_width))  # 计算缩放比例
                batch_widths[i] = img_width * this_scale  # 计算缩放后的宽度
                batch_heights[i] = img_height * this_scale  # 计算缩放后的高度

            batch_width = np.max(batch_widths)  # 获取最大宽度
            batch_height = np.max(batch_heights)  # 获取最大高度

        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))  # 调整宽度到padding常数的倍数
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))  # 调整高度到padding常数的倍数

        return batch_width, batch_height

    def __getitem__(self, index):
        # 获取单个样本函数
        this_record = self.list_sample[index]  # 获取当前样本记录
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])  # 构建图像路径
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])  # 构建分割标签路径

        img = Image.open(image_path).convert('RGB')  # 打开并转换图像为RGB格式
        segm = Image.open(segm_path).convert('L')  # 打开并转换分割标签为灰度图

        # data augmentation
        img = np.array(img)  # 转换图像为numpy数组
        segm = np.array(segm)  # 转换分割标签为numpy数组
        for seq in self.aug_pipe:  # 应用数据增强管道
            img, segm = imgaug_mask(img, segm, seq)

        output = dict()  # 创建输出字典
        output['image'] = img  # 存储图像数据
        output['mask'] = segm  # 存储分割标签

        return output

    def collate_fn(self, batch):
        # 批次整理函数，用于PyTorch DataLoader
        batch_width, batch_height = self.get_batch_size(batch)  # 获取批次尺寸
        out = {}  # 初始化输出字典
        images = []  # 初始化图像列表
        masks = []  # 初始化掩码列表
        raw_images = []  # 初始化原始图像列表

        for item in batch:  # 遍历批次中的每个样本
            img = deepcopy(item['image'])  # 深拷贝图像数据
            segm = item['mask']  # 获取分割标签

            img = Image.fromarray(img)  # 将numpy数组转换为图像
            segm = Image.fromarray(segm)  # 将numpy数组转换为图像

            img = self.resize_padding(img, (batch_width, batch_height))  # 调整图像尺寸并填充
            img = self.img_transform(img)  # 图像标准化处理
            segm = self.resize_padding(segm, (batch_width, batch_height), Image.NEAREST)  # 调整分割标签尺寸并填充
            segm = segm.resize(
                (batch_width // self.segm_downsampling_rate, batch_height // self.segm_downsampling_rate),
                Image.NEAREST)  # 调整分割标签到网络输出尺寸

            images.append(torch.from_numpy(img).float())  # 添加处理后的图像到列表
            masks.append(torch.from_numpy(np.array(segm)).long())  # 添加处理后的分割标签到列表
            raw_images.append(item['image'])  # 添加原始图像到列表

        out['images'] = torch.stack(images)  # 堆叠图像张量
        out['masks'] = torch.stack(masks)  # 堆叠分割标签张量
        out['raw_img'] = raw_images  # 存储原始图像
        return out

    def __len__(self):
        # 获取数据集长度函数
        return self.num_sample


class LaneDetec(ADE200kDataset):
    # 车道线检测数据集类，继承自ADE20kDataset
    def __init__(self, odgt, opt, dynamic_batchHW=False, **kwargs):
        super(LaneDetec, self).__init__(odgt, opt, dynamic_batchHW, **kwargs)  # 调用父类初始化

    def __getitem__(self, index):
        # 获取单个样本函数，重写了父类方法
        this_record = self.list_sample[index]  # 获取当前样本记录
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])  # 构建图像路径
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])  # 构建分割标签路径

        img = Image.open(image_path).convert('RGB')  # 打开并转换图像为RGB格式
        segm = Image.open(segm_path).convert('L')  # 打开并转换分割标签为灰度图

        # data augmentation
        img = np.array(img)[800:, :, :]  # 移除图片上方的天空部分      # 裁剪掉图像上方800像素（天空部分）
        segm = np.array(segm)[800:, :]  # 裁剪对应的分割标签
        for seq in self.aug_pipe:  # 应用数据增强管道
            img, segm = imgaug_mask(img, segm, seq)

        output = dict()  # 创建输出字典
        output['image'] = img  # 存储图像数据
        output['mask'] = segm  # 存储分割标签

        return output


# 用于nuImages数据集的Dataset类
class NuImagesDataset(ADE200kDataset):
    # NuImages数据集类，继承自ADE20kDataset
    def __init__(self, data_root, opt, version='v1.0-train', **kwargs):
        super(NuImagesDataset, self).__init__(None, opt, **kwargs)  # 调用父类初始化，不传入odgt
        self.nuim = NuImages(dataroot=data_root, version=version, lazy=False)  # 初始化NuImages对象
        max_sample = kwargs.get('max_sample', -1)  # 获取最大样本数参数
        if max_sample > 0:  # 如果设置了最大样本数
            # 限制样本数量
            self.nuim.sample = self.nuim.sample[:max_sample]  # 限制样本数量
        else:
            self.nuim.sample = self.nuim.sample  # 使用全部样本
        self.num_sample = len(self.nuim.sample)  # 获取样本总数
        print(f'Load {self.num_sample} samples from {version}')  # 打印加载的样本数量

    def __getitem__(self, index):
        # 获取单个样本函数，重写了父类方法
        sample = self.nuim.sample[index]  # 获取样本 这里是从文件中获取数据
        sd_token = sample['key_camera_token']  # 获取相机token
        sample_data = self.nuim.get('sample_data', sd_token)  # 获取样本数据
        # todo 用opencv进行加速替换
        im_path = os.path.join(self.nuim.dataroot, sample_data['filename'])  # 构建图像路径
        img = Image.open(im_path).convert('RGB')  # 打开并转换图像为RGB格式  这里使用numpy 加速
        img = np.array(img)  # 转换为numpy数组

        semseg_mask, instanceseg_mask = self.nuim.get_segmentation(sd_token)  # 获取语义分割和实例分割标签

        semseg_mask[semseg_mask == 31] = 0  # 31是vehicle.ego, 不做预测  # 将vehicle.ego类别标记为背景
        output = dict()  # 创建输出字典
        output['image'] = img  # 存储图像数据
        output['mask'] = semseg_mask  # 存储语义分割标签
        output['ins_mask'] = instanceseg_mask  # 存储实例分割标签
        # self.nuim.render_image(sd_token, annotation_type='all', with_category=True, with_attributes=True, out_path='/home/dataset/nuImages/ImageData/out_test.png')
        return output

    def oepncv_fast(self, im_path):
        # 使用 OpenCV 读取图像
        img_cv = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError(f"无法读取图像: {im_path}")

        # 将 BGR 转换为 RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # 将 numpy 数组转换回 PIL Image 对象
        img = Image.fromarray(img_cv)
        return img

    def __len__(self):
        # 获取数据集长度函数
        return self.num_sample