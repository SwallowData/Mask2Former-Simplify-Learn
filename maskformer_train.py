#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   maskformer3D.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   DeformTransAtten分割网络训练代码
'''

# here put the import lib

# 导入必要的库
from statistics import mean  # 用于计算平均值
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import os  # 操作系统接口
import time  # 时间处理
import datetime  # 日期时间处理
from torch import nn  # 神经网络模块
from torch.nn import functional as F  # 函数接口
import torch.optim as optim  # 优化器
from torch import distributed as dist  # 分布式训练
from torch.utils.data import DataLoader, SubsetRandomSampler  # 数据加载器
import sys  # 系统相关功能
import math  # 数学函数
import itertools  # 迭代器工具
from PIL import Image  # 图像处理
import wandb  # Weights & Biases实验跟踪工具

# 导入自定义模块
from modeling.MaskFormerModel import MaskFormerModel  # MaskFormer模型定义
from utils.criterion import SetCriterion, Criterion  # 损失函数
from utils.matcher import HungarianMatcher  # 匈牙利匹配算法
from utils.summary import create_summary  # 摘要记录工具
from utils.solver import maybe_add_gradient_clipping  # 梯度裁剪工具
from utils.misc import load_parallal_model  # 模型加载工具
from dataset.NuImages import NuImages  # NuImages数据集处理
from Segmentation import Segmentation  # 分割处理类


class MaskFormer():
    """
    MaskFormer训练主类
    负责模型初始化、训练、评估等核心功能
    """

    def __init__(self, cfg):
        # 调用父类初始化方法
        super().__init__()
        # 保存配置参数
        self.cfg = cfg
        # 获取查询数量配置
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # 获取尺寸可分性配置
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        # 获取类别数量配置
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # 设置设备(CUDA)
        self.device = torch.device("cuda", cfg.local_rank)
        # 获取训练状态配置
        self.is_training = cfg.MODEL.IS_TRAINING
        # 获取批处理大小配置
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        # 获取学习率配置
        self.last_lr = cfg.SOLVER.LR
        # 设置起始训练轮次
        self.start_epoch = 0

        # 初始化MaskFormer模型
        self.model = MaskFormerModel(cfg)
        # 如果配置了预训练权重且文件存在，则加载预训练模型
        if cfg.MODEL.PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.PRETRAINED_WEIGHTS):
            self.load_model(cfg.MODEL.PRETRAINED_WEIGHTS, cfg.MODEL.AUTOSELECT)
            print("loaded pretrain mode:{}".format(cfg.MODEL.PRETRAINED_WEIGHTS))

        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        # 如果使用多GPU，则使用分布式数据并行
        if cfg.ngpus > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank],
                                                             output_device=cfg.local_rank)

            # 初始化训练相关参数
        self._training_init(cfg)

        # 创建WandB运行名称（基于当前时间）
        run_name = datetime.datetime.now().strftime("swin-%Y-%m-%d-%H-%M")
        # 初始化WandB实验跟踪
        self.run = wandb.init(
            project=cfg.project_name,
            name=run_name,
            settings=wandb.Settings(init_timeout=120)
        )

        # 监控模型
        wandb.watch(self.model)

    def build_optimizer(self):
        """
        构建优化器，支持梯度裁剪
        """

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            # 获取梯度裁剪值配置
            clip_norm_val = self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            # 判断是否启用全模型梯度裁剪
            enable = (
                    self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            # 定义全模型梯度裁剪优化器类
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    # 获取所有参数
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    # 执行梯度裁剪
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    # 调用父类step方法
                    super().step(closure=closure)

            # 根据是否启用返回相应的优化器类
            return FullModelGradientClippingOptimizer if enable else optim

        # 获取优化器类型配置
        optimizer_type = self.cfg.SOLVER.OPTIMIZER
        # 根据配置选择优化器类型
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                self.model.parameters(), self.last_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                self.model.parameters(), self.last_lr)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        # 如果不是全模型梯度裁剪，则使用普通梯度裁剪
        if not self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(self.cfg, optimizer)

        return optimizer

    def load_model(self, pretrain_weights, auto):
        """
        加载预训练模型权重
        """
        # 从指定路径加载模型状态字典
        bast_pertrain_weights = pretrain_weights
        if auto:
            bast_pertrain_weights = self.select_bast_model(pretrain_weights)
        state_dict = torch.load(bast_pertrain_weights, map_location='cuda:0')
        print('loaded pretrained weights form %s !' % bast_pertrain_weights)

        # 获取模型权重
        ckpt_dict = state_dict['model']
        # 设置学习率（注释掉的代码显示从checkpoint加载）
        self.last_lr = state_dict['lr']  # state_dict['lr']
        # 设置起始轮次（注释掉的代码显示从checkpoint加载）
        self.start_epoch = state_dict['epoch']  # state_dict['epoch']
        # 加载并行模型权重
        self.model = load_parallal_model(self.model, ckpt_dict)

    def select_bast_model(self, base_path):
        root_dict = base_path.split("/mas")[0]
        best_model = os.path.join(root_dict, 'best_model') if os.path.exists(
            os.path.join(root_dict, 'best_model')) else os.mkdir(root_dict)
        finally_model = os.path.join(root_dict, 'finally_model') if os.path.exists(
            os.path.join(root_dict, 'best_model')) else os.mkdir(root_dict)
        all_model_list = [os.path.join(best_model, i) for i in os.listdir(best_model)] + [os.path.join(finally_model, i)
                                                                                          for i in
                                                                                          os.listdir(finally_model)]
        max_info = {'score': 0, 'file': ''}
        if all_model_list:
            for model in all_model_list:
                score = float(model.split("dice")[-1].replace(".pth", ""))
                if score > max_info['score']:
                    max_info['score'] = score
                    max_info['file'] = os.path.join(model)
        else:
            return base_path
        return max_info['file']

    def _training_init(self, cfg):
        """
        初始化训练相关参数
        """
        # Loss parameters:
        # 获取深度监督配置
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        # 获取无对象权重配置
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        # 获取各类损失权重配置
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        # building criterion
        # 构建匈牙利匹配器
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        # 定义权重字典
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        # 如果启用深度监督，则添加辅助权重
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # 定义损失类型
        losses = ["labels", "masks"]
        # 构建损失函数准则
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )

        # 创建摘要写入器
        self.summary_writer = create_summary(0, log_dir=cfg.TRAIN.LOG_DIR)
        # 设置模型保存文件夹
        self.save_folder = cfg.TRAIN.CKPT_DIR
        # 构建优化器
        self.optim = self.build_optimizer()
        # 创建学习率调度器（基于评估分数调整学习率）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.9, patience=10)

    def reduce_mean(self, tensor, nprocs):
        """
        用于平均所有gpu上的运行结果，比如loss
        """
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def train(self, train_sampler, data_loader, eval_loder, n_epochs):
        """
        主训练循环函数
        """
        # 设置最佳评估分数初始值
        max_score = 0.88
        # 自动小批量训练
        if self.start_epoch >= n_epochs - 1:
            n_epochs = self.start_epoch + 5
        print('self.start_epoch:', self.start_epoch, '\tn_epochs:', n_epochs)
        # 遍历训练轮次
        for epoch in range(self.start_epoch + 1, n_epochs):
            # 如果使用采样器，则设置当前轮次
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # 执行一个训练轮次
            train_loss = self.train_epoch(data_loader, epoch)
            # 执行评估
            evaluator_score = self.evaluate(eval_loder)
            # 生成评估样本
            evaluator_samples = self.evaluate_sample()
            # 根据评估分数调整学习率
            self.scheduler.step(evaluator_score)
            # 使用WandB记录日志
            wandb.log({
                "evaluator_score": evaluator_score,
                "train_loss": train_loss,
                "samples": [wandb.Image(sample) for sample in evaluator_samples],
            })
            print('evaluator_score:', evaluator_score, '\tmax_score:', max_score)

            # 如果当前评估分数超过最佳分数，则保存模型
            if evaluator_score > max_score:
                max_score = evaluator_score  # todo 这里发生了修改
                # 构造模型保存路径
                ckpt_path = os.path.join(os.path.join(self.save_folder, 'best_model'),
                                         'mask2former_Epoch{0}_dice{1:.4f}.pth'.format(epoch, max_score))
                # 构造保存状态字典
                save_state = {'model': self.model.state_dict(),
                              'lr': self.optim.param_groups[0]['lr'],
                              'epoch': epoch}
                # 保存模型
                torch.save(save_state, ckpt_path)
                print('best weights {0} saved success!'.format(ckpt_path))
            if epoch == (n_epochs - 1):
                max_score = evaluator_score  # todo 这里发生了修改
                # 构造模型保存路径
                ckpt_path = os.path.join(os.path.join(self.save_folder, 'finally_model'),
                                         'mask2former_Epoch{0}_dice{1:.4f}.pth'.format(epoch, max_score))
                # 构造保存状态字典
                save_state = {'model': self.model.state_dict(),
                              'lr': self.optim.param_groups[0]['lr'],
                              'epoch': epoch}
                # 保存模型
                torch.save(save_state, ckpt_path)
                print('finally weights {0} saved success!'.format(ckpt_path))
        # 关闭摘要写入器
        self.summary_writer.close()

    def train_epoch(self, data_loader, epoch):
        """
        单个训练轮次执行函数
        """
        # 设置模型为训练模式
        self.model.train()
        # 设置损失函数为训练模式
        self.criterion.train()
        # 记录开始时间
        load_t0 = time.time()
        # 初始化各类损失列表
        losses_list = []
        loss_ce_list = []
        loss_dice_list = []
        loss_mask_list = []

        # 遍历数据加载器中的每个批次
        for i, batch in enumerate(data_loader):
            # 将输入数据移动到指定设备
            inputs = batch['images'].to(device=self.device, non_blocking=True)
            # 获取目标数据
            targets = batch['masks']

            # 前向传播 完前向传播
            outputs = self.model(inputs)
            # 计算损失
            losses = self.criterion(outputs, targets)
            # 获取权重字典
            weight_dict = self.criterion.weight_dict

            # 初始化各类损失值
            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            # 遍历所有损失项
            for k in list(losses.keys()):
                if k in weight_dict:
                    # 根据权重字典加权损失
                    losses[k] *= self.criterion.weight_dict[k]
                    # 分类累加各类损失
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif '_dice' in k:
                        loss_dice += losses[k]
                    elif '_mask' in k:
                        loss_mask += losses[k]
                else:
                    # 如果不在权重字典中则移除该损失项
                    losses.pop(k)
            # 计算总损失
            loss = loss_ce + loss_dice + loss_mask
            # 不计算梯度的情况下记录损失值
            with torch.no_grad():
                losses_list.append(loss.item())
                loss_ce_list.append(loss_ce.item())
                loss_dice_list.append(loss_dice.item())
                loss_mask_list.append(loss_mask.item())

            # 清零梯度
            self.model.zero_grad()
            self.criterion.zero_grad()
            # 反向传播
            loss.backward()
            # 执行优化器步骤 这里面自定义实现optim里面实现梯度裁剪
            self.optim.step()

            # 计算已用时间和预计剩余时间
            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            # 获取当前学习率
            curent_lr = self.optim.param_groups[0]['lr']
            # 构造进度显示字符串
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f} loss_mask:{(np.mean(loss_mask_list)):.6f}, lr:{curent_lr:.2e} '
            # 打印进度信息
            print(progress, end=' ')
            sys.stdout.flush()

            # 记录损失到摘要写入器
        self.summary_writer.add_scalar('loss', loss.item(), epoch)
        # 返回最终损失值
        return loss.item()

    @torch.no_grad()
    def evaluate(self, eval_loder):
        """
        模型评估函数
        """
        # 设置模型为评估模式
        self.model.eval()
        # 初始化dice分数列表
        dice_score = []

        # 遍历评估数据加载器
        for batch in eval_loder:
            # 将输入张量移动到指定设备
            inpurt_tensor = batch['images'].to(device=self.device, non_blocking=True)
            # 获取真实标签掩码
            gt_mask = batch['masks'][0]

            # 前向传播
            outputs = self.model(inpurt_tensor)
            # 获取分类结果和掩码预测结果
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            # 执行语义推理
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)
            # 获取预测掩码
            pred_mask = pred_masks[0]
            # 获取二值化的真实掩码
            gt_binary_mask = self._get_binary_mask(gt_mask)
            # 计算dice分数
            dice = self._get_dice(pred_mask, gt_binary_mask.to(self.device))
            # 添加到分数列表
            dice_score.append(dice.item())
        # 计算平均dice分数
        score = np.mean(dice_score)
        print('evaluate dice: {0}'.format(score))
        # 返回评估分数
        return score

    @torch.no_grad()
    def evaluate_sample(self):
        """
        生成评估样本可视化结果
        """
        # 初始化NuImages数据集
        nuim = NuImages(dataroot=self.cfg.DATASETS.ROOT_DIR, version='v1.0-test')  # v1.0-test or v1.0-mini
        # 随机选择样本索引
        sample_idx_list = np.random.choice(len(nuim.sample), 10, replace=False)
        # 初始化分割处理器
        seg_handler = Segmentation(self.cfg, self.model)
        # 初始化输入图像和渲染图像列表
        input_imgs = []
        render_imgs = []
        # 遍历样本索引
        for idx in sample_idx_list:
            sample = nuim.sample[idx]
            sd_token = sample['key_camera_token']
            sample_data = nuim.get('sample_data', sd_token)

            # 获取图像路径
            im_path = os.path.join(nuim.dataroot, sample_data['filename'])
            input_imgs.append(im_path)
        # 执行前向推理
        preds = seg_handler.forward(input_imgs)
        # 生成渲染图像
        for i, img_path in enumerate(input_imgs):
            img = Image.open(img_path)
            render_img = nuim.render_predict(img, preds[i])
            render_imgs.append(render_img)
        # 返回渲染图像列表
        return render_imgs

    def _get_dice(self, predict, target):
        """
        计算dice系数
        """
        # 设置平滑值防止除零错误
        smooth = 1e-5
        # 重塑预测张量
        predict = predict.contiguous().view(predict.shape[0], -1)
        # 重塑目标张量
        target = target.contiguous().view(target.shape[0], -1)

        # 计算分子部分（预测和目标的点积）
        num = torch.sum(torch.mul(predict, target), dim=1)
        # 计算分母部分（预测和目标的和）
        den = predict.sum(-1) + target.sum(-1)
        # 计算dice分数
        score = (2 * num + smooth).sum(-1) / (den + smooth).sum(-1)
        # 返回平均分数
        return score.mean()

    def _get_binary_mask(self, target):
        """
        将目标转换为每类的二值掩码
        """
        # 获取目标尺寸
        y, x = target.size()
        # 创建one-hot编码张量
        target_onehot = torch.zeros(self.num_classes + 1, y, x)
        # 使用scatter函数填充one-hot编码
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        # 返回除背景外的类别掩码
        return target_onehot[1:]

    def semantic_inference(self, mask_cls, mask_pred):
        """
        执行语义分割推理
        """
        # 对分类结果应用softmax并去除背景类
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        # 对掩码预测结果应用sigmoid
        mask_pred = mask_pred.sigmoid()
        # 使用爱因斯坦求和约定计算语义分割结果
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        # 返回语义分割结果
        return semseg

    # 实例分割待调试
    def instance_inference(self, mask_cls, mask_pred):
        """
        执行实例分割推理（待调试）
        """
        # mask_pred is already processed to have the same shape as original input
        # 获取图像尺寸
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        # 对分类结果应用softmax并去除最后一类
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # 创建标签张量
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                                                                     1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        # 获取top-k分数和索引
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        # 获取对应标签
        labels_per_image = labels[topk_indices]

        # 计算topk索引
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        # 根据索引获取掩码预测
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        # 如果是全景分割，则只保留"thing"类别
        if self.panoptic_on:
            # 初始化保留标记
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            # 过滤分数、标签和掩码预测
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        # 创建实例对象
        result = Instances(image_size)
        # mask (before sigmoid)
        # 设置预测掩码（二值化）
        result.pred_masks = (mask_pred > 0).float()
        # 初始化预测边界框
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        # 计算平均掩码概率
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                    result.pred_masks.flatten(1).sum(1) + 1e-6)
        # 设置最终分数
        result.scores = scores_per_image * mask_scores_per_image
        # 设置预测类别
        result.pred_classes = labels_per_image
        # 返回结果
        return result