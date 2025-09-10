#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/10/11 19:54:03
@Author  :   zzubqh
@Version :   1.0
@Contact :   baiqh@microport.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

# 导入argparse模块用于命令行参数解析
import argparse
# 导入os模块用于操作系统相关功能
import os

# 设置CUDA可见设备为0-7号GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  #

# 从fvcore.common.config导入CfgNode类
from fvcore.common.config import CfgNode
# 从configs.config导入Config类
from configs.config import Config
# 导入torch模块
import torch
# 从maskformer_train导入MaskFormer类
from maskformer_train import MaskFormer
# 从dataset.dataset导入ADE200kDataset和NuImagesDataset类
from dataset.dataset import ADE200kDataset, NuImagesDataset
# 从Segmentation导入Segmentation类
from Segmentation import Segmentation

import os

# os.environ["WANDB_MODE"] = "offline"

os.environ["WANDB_API_KEY"] = "local-ea"
# 如果CUDA设备数量大于1，则初始化分布式训练进程组
if torch.cuda.device_count() > 1:
    # 使用nccl后端初始化分布式训练进程组
    torch.distributed.init_process_group(backend='nccl')


# 定义用户分散数据收集函数
def user_scattered_collate(batch):
    # 从batch中提取图像数据
    data = [item['images'] for item in batch]
    # 从batch中提取掩码数据
    masks = [item['masks'] for item in batch]
    # 构造输出字典，将图像和掩码数据合并
    out = {'images': torch.cat(data, dim=0), 'masks': torch.cat(masks, dim=0)}
    # 返回合并后的数据
    return out


# 定义获取命令行参数函数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加配置文件路径参数，默认为configs/maskformer_nuimages.yaml
    parser.add_argument('--config', type=str, default='configs/maskformer_nuimages.yaml')
    # 添加本地rank参数，默认为0
    parser.add_argument('--local_rank', type=int, default=0)
    # 添加GPU数量参数，默认为1
    parser.add_argument("--ngpus", default=1, type=int)
    # 添加项目名称参数，默认为NuImages_swin_base_Seg
    parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)

    parser.add_argument("--sumge", default='', type=str)

    # 解析命令行参数
    args = parser.parse_args()
    # 从配置文件创建Config对象
    cfg_ake150 = Config.fromfile(args.config)

    # 从yaml文件加载配置，允许不安全操作 这里是将_BASE_这里的关键字加载 这种双重加载机制确保了配置系统的灵活性和兼容性。
    cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True)
    # 更新配置对象的属性
    cfg_base.update(cfg_ake150.__dict__.items())

    # 将cfg_base赋值给cfg
    cfg = cfg_base
    # 遍历args的属性并更新到cfg中
    for k, v in args.__dict__.items():
        cfg[k] = v

    # 将cfg转换为Config对象
    cfg = Config(cfg)

    # 设置GPU数量为当前CUDA设备数量
    cfg.ngpus = torch.cuda.device_count()
    # 如果CUDA设备数量大于1
    if torch.cuda.device_count() > 1:
        # 获取当前进程的rank并设置到cfg中
        cfg.local_rank = torch.distributed.get_rank()
        # 根据local_rank设置当前CUDA设备
        torch.cuda.set_device(cfg.local_rank)
    # 返回配置对象
    return cfg


# 定义训练ADE200k数据集函数
def train_ade200k():
    # 获取配置参数
    cfg = get_args()
    # 创建ADE200k训练数据集对象，启用动态batchHW
    dataset_train = ADE200kDataset(cfg.DATASETS.TRAIN, cfg, dynamic_batchHW=True)
    # 如果GPU数量大于1
    if cfg.ngpus > 1:
        # 创建分布式训练采样器
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, rank=cfg.local_rank)
    else:
        # 否则采样器为None
        train_sampler = None
        # 创建训练数据加载器
    loader_train = torch.utils.data.DataLoader(
        # 指定数据集
        dataset_train,
        # 指定批次大小
        batch_size=cfg.TRAIN.BATCH_SIZE,
        # 根据采样器是否存在决定是否打乱数据
        shuffle=False if train_sampler is not None else True,
        # 指定数据收集函数
        collate_fn=dataset_train.collate_fn,
        # 指定工作进程数
        num_workers=cfg.TRAIN.WORKERS,
        # 是否丢弃最后一个不完整的批次
        drop_last=True,
        # 是否将数据加载到锁页内存
        pin_memory=True,
        # 指定采样器
        sampler=train_sampler)

    # 创建ADE200k验证数据集对象
    dataset_eval = ADE200kDataset(cfg.DATASETS.VALID, cfg)
    # 创建验证数据加载器
    loader_eval = torch.utils.data.DataLoader(
        # 指定数据集
        dataset_eval,
        # 批次大小为1
        batch_size=1,
        # 不打乱数据
        shuffle=False,
        # 指定数据收集函数
        collate_fn=dataset_eval.collate_fn,
        # 指定工作进程数
        num_workers=cfg.TRAIN.WORKERS)

    # 创建MaskFormer模型对象
    seg_model = MaskFormer(cfg)
    # 调用模型的训练方法
    seg_model.train(train_sampler, loader_train, loader_eval, cfg.TRAIN.EPOCH)


# 定义训练NuImages数据集函数
def train_nuimages():
    # 获取配置参数
    cfg = get_args()
    # 创建NuImages训练数据集对象，使用v1.0-train版本
    dataset_train = NuImagesDataset(cfg.DATASETS.ROOT_DIR, cfg, version='v1.0-train',
                                    max_sample=1000)  # v1.0-mini or v1.0-train
    # 创建NuImages验证数据集对象，使用v1.0-val版本
    dataset_eval = NuImagesDataset(cfg.DATASETS.ROOT_DIR, cfg, version='v1.0-val',
                                   max_sample=100)  # v1.0-mini or v1.0-val

    # 如果GPU数量大于1
    if cfg.ngpus > 1:
        # 创建训练数据分布式采样器
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, rank=cfg.local_rank)
        # 创建验证数据分布式采样器
        eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval, rank=cfg.local_rank)
    else:
        # 否则采样器为None
        train_sampler = None
        eval_sampler = None

    # 创建训练数据加载器
    loader_train = torch.utils.data.DataLoader(
        # 指定数据集
        dataset_train,
        # 指定批次大小
        batch_size=cfg.TRAIN.BATCH_SIZE,
        # 根据采样器是否存在决定是否打乱数据
        shuffle=False if train_sampler is not None else True,
        # 指定数据收集函数
        collate_fn=dataset_train.collate_fn,
        # 指定工作进程数
        num_workers=cfg.TRAIN.WORKERS,
        # 是否丢弃最后一个不完整的批次
        drop_last=True,
        # 是否将数据加载到锁页内存
        pin_memory=True,
        # 指定采样器 这里的采样器是空的
        sampler=train_sampler
    )

    # 创建验证数据加载器
    loader_eval = torch.utils.data.DataLoader(
        # 指定数据集
        dataset_eval,
        # 批次大小为1
        batch_size=1,
        # 根据采样器是否存在决定是否打乱数据
        shuffle=False if eval_sampler is not None else True,
        # 指定数据收集函数
        collate_fn=dataset_eval.collate_fn,
        # 指定工作进程数
        num_workers=cfg.TRAIN.WORKERS,
        # 不丢弃最后一个不完整的批次
        drop_last=False,
        # 是否将数据加载到锁页内存
        pin_memory=True,
        # 指定采样器
        sampler=eval_sampler)

    # 创建MaskFormer模型对象
    seg_model = MaskFormer(cfg)
    # 调用模型的训练方法
    seg_model.train(train_sampler, loader_train, loader_eval, cfg.TRAIN.EPOCH)


# 定义分割测试函数
def segmentation_test():
    # 获取配置参数
    cfg = get_args()
    # 创建分割处理对象
    segmentation_handler = Segmentation(cfg)
    # 调用处理对象的forward方法
    segmentation_handler.forward()


# 程序入口点
if __name__ == '__main__':
    # 调用训练NuImages数据集函数
    train_nuimages()
    # 调用分割测试函数（被注释掉）
    # segmentation_test()