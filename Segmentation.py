# 导入统计模块，用于统计信息
from statistics import mode
# 导入fvcore的配置节点类，用于处理配置
from fvcore.common.config import CfgNode
# 导入numpy用于数值计算
import numpy as np
# 导入操作系统接口模块
import os
# 导入OpenCV用于图像处理
import cv2
# 导入glob模块用于文件路径匹配
import glob
# 导入tqdm用于显示进度条
import tqdm
# 从PIL导入Image模块用于图像处理
from PIL import Image
# 从PIL导入ImageOps模块用于图像操作
from PIL import ImageOps
# 导入PyTorch核心模块
import torch
# 从torch导入神经网络模块
from torch import nn
# 从torch.nn导入函数式接口
from torch.nn import functional as F
# 从modeling目录导入MaskFormerModel模型
from modeling.MaskFormerModel import MaskFormerModel
# 从utils.misc导入工具函数
from utils.misc import load_parallal_model
# 从utils.misc导入可视化类
from utils.misc import ADEVisualize
# 从configs.config导入配置类
from configs.config import Config
# 导入argparse用于命令行参数解析
import argparse


# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog

# 定义分割类
class Segmentation():
    # 初始化函数
    def __init__(self, cfg, model=None):
        # 保存配置参数
        self.cfg = cfg
        # 获取模型中查询数量配置
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # 获取模型尺寸可整除性配置
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        # 获取语义分割头类别数配置
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # 设置设备为CUDA设备
        self.device = torch.device("cuda", cfg.local_rank)

        # 数据处理程序 data processing program
        # 设置填充常数，resnet总共下采样5次
        self.padding_constant = 2 ** 5
        # 设置测试目录
        self.test_dir = cfg.TEST.TEST_DIR
        # 设置输出目录
        self.output_dir = cfg.TEST.SAVE_DIR
        # 设置图像最大尺寸
        self.imgMaxSize = cfg.INPUT.CROP.MAX_SIZE
        # 设置像素均值
        self.pixel_mean = np.array(cfg.DATASETS.PIXEL_MEAN)
        # 设置像素标准差
        self.pixel_std = np.array(cfg.DATASETS.PIXEL_STD)
        # 初始化可视化对象
        self.visualize = ADEVisualize()
        # 初始化模型为None
        self.model = None

        # 获取预训练权重路径
        pretrain_weights = cfg.MODEL.PRETRAINED_WEIGHTS
        # 如果传入了模型实例
        if model is not None:
            # 直接使用传入的模型
            self.model = model
        # 如果预训练权重文件存在
        elif os.path.exists(pretrain_weights):
            # 创建MaskFormerModel实例
            self.model = MaskFormerModel(cfg, )
            # 加载模型权重
            self.load_model(pretrain_weights)
        else:
            # 打印错误信息，提示检查权重文件
            print(f'please check weights file: {cfg.MODEL.PRETRAINED_WEIGHTS}')

    # 加载模型权重函数
    def load_model(self, pretrain_weights):
        # 从文件加载状态字典
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')

        # 获取模型状态字典
        ckpt_dict = state_dict['model']
        # 获取学习率
        self.last_lr = state_dict['lr']
        # 获取起始epoch
        self.start_epoch = state_dict['epoch']
        # 加载并行模型
        self.model = load_parallal_model(self.model, ckpt_dict)
        # 将模型移到指定设备
        self.model = self.model.to(self.device)
        # 设置模型为评估模式
        self.model.eval()
        # 打印加载的预训练模型信息
        print("loaded pretrain mode:{}".format(pretrain_weights))

    # 图像变换函数
    def img_transform(self, img):
        # 0-255 to 0-1 将图像像素值从0-255范围转换到0-1范围
        img = np.float32(np.array(img)) / 255.
        # 进行标准化处理
        img = (img - self.pixel_mean) / self.pixel_std
        # 转换通道顺序从HWC到CHW
        img = img.transpose((2, 0, 1))
        # 返回处理后的图像
        return img

    # 将x舍入到最接近的p的倍数且x' >= x  Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        # 计算并返回最接近的倍数
        return ((x - 1) // p + 1) * p

    # 获取图像缩放比例
    def get_img_ratio(self, img_size, target_size):
        # 计算图像长宽比
        img_rate = np.max(img_size) / np.min(img_size)
        # 计算目标长宽比
        target_rate = np.max(target_size) / np.min(target_size)
        # 如果图像长宽比大于目标长宽比
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            # 按短边缩放
            ratio = min(target_size) / min(img_size)
        # 返回缩放比例
        return ratio

    # 调整图像大小并填充
    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        # 获取图像宽高
        w, h = img.size
        # 获取目标宽高
        target_w, target_h = outsize[0], outsize[1]
        # 获取缩放比例
        ratio = self.get_img_ratio([w, h], outsize)
        # 计算缩放后的宽高
        ow, oh = round(w * ratio), round(h * ratio)
        # 调整图像大小
        img = img.resize((ow, oh), Interpolation)
        # 计算需要填充的高度和宽度
        dh, dw = target_h - oh, target_w - ow
        # 计算上下左右填充像素数
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        # 对图像进行填充，左顶右底顺时针
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
        # 返回处理后的图像和填充信息
        return img, [left, top, right, bottom]

    # 获取图像缩放比例（重复定义）
    def get_img_ratio(self, img_size, target_size):
        # 计算图像长宽比
        img_rate = np.max(img_size) / np.min(img_size)
        # 计算目标长宽比
        target_rate = np.max(target_size) / np.min(target_size)
        # 如果图像长宽比大于目标长宽比
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            # 按短边缩放
            ratio = min(target_size) / min(img_size)
        # 返回缩放比例
        return ratio

    # 图像预处理函数
    def image_preprocess(self, img):
        # 获取图像高度和宽度
        img_height, img_width = img.shape[0], img.shape[1]
        # 获取缩放比例
        this_scale = self.get_img_ratio((img_width, img_height),
                                        self.imgMaxSize)  # self.imgMaxSize / max(img_height, img_width)
        # 计算目标宽度
        target_width = img_width * this_scale
        # 计算目标高度
        target_height = img_height * this_scale
        # 计算输入宽度并舍入到最近的填充常数倍数
        input_width = int(self.round2nearest_multiple(target_width, self.padding_constant))
        # 计算输入高度并舍入到最近的填充常数倍数
        input_height = int(self.round2nearest_multiple(target_height, self.padding_constant))

        # 调整图像大小并填充
        img, padding_info = self.resize_padding(Image.fromarray(img), (input_width, input_height))
        # 对图像进行变换
        img = self.img_transform(img)

        # 构建变换信息字典
        transformer_info = {'padding_info': padding_info, 'scale': this_scale,
                            'input_size': (input_height, input_width)}
        # 将图像转换为张量并移到指定设备
        input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
        # 返回输入张量和变换信息
        return input_tensor, transformer_info

    # 语义推理函数
    def semantic_inference(self, mask_cls, mask_pred):
        # 对mask_cls进行softmax处理并去掉第一列（第一列是背景）
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]
        # 对mask_pred进行sigmoid处理
        mask_pred = mask_pred.sigmoid()
        # 通过爱因斯坦求和进行语义分割计算
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        # 返回计算结果并转换为numpy数组
        return semseg.cpu().numpy()

    # 后处理函数
    def postprocess(self, pred_mask, transformer_info, target_size):
        # 获取预测掩码的高度和宽度
        oh, ow = pred_mask.shape[0], pred_mask.shape[1]
        # 获取填充信息
        padding_info = transformer_info['padding_info']

        # 获取左上右下填充像素数
        left, top, right, bottom = padding_info[0], padding_info[1], padding_info[2], padding_info[3]
        # 去除填充部分
        mask = pred_mask[top: oh - bottom, left: ow - right]
        # 调整图像大小到目标尺寸
        mask = cv2.resize(mask.astype(np.uint8), dsize=target_size, interpolation=cv2.INTER_NEAREST)
        # 返回处理后的掩码
        return mask

    # 前向传播函数，使用torch.no_grad()装饰器不计算梯度
    @torch.no_grad()
    def forward(self, img_list=None):
        # 如果图像列表为空或长度为0
        if img_list is None or len(img_list) == 0:
            # 使用glob获取测试目录下的所有图片文件
            img_list = glob.glob(self.test_dir + '/*.[jp][pn]g')
        # 初始化掩码图像列表
        mask_images = []
        # 遍历图像路径列表，使用tqdm显示进度条
        for image_path in tqdm.tqdm(img_list):
            # img_name = os.path.basename(image_path)
            # seg_name = img_name.split('.')[0] + '_seg.png'
            # output_path = os.path.join(self.output_dir, seg_name)
            # 打开图像并转换为RGB模式
            img = Image.open(image_path).convert('RGB')
            # 获取图像高度和宽度
            img_height, img_width = img.size[1], img.size[0]
            # 对图像进行预处理
            inpurt_tensor, transformer_info = self.image_preprocess(np.array(img))

            # 通过模型获取输出
            outputs = self.model(inpurt_tensor)
            # 获取预测的logits
            mask_cls_results = outputs["pred_logits"]
            # 获取预测的掩码
            mask_pred_results = outputs["pred_masks"]

            # 对预测掩码进行插值处理
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(inpurt_tensor.shape[-2], inpurt_tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            # 进行语义推理
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)
            # 获取预测掩码中最大概率的类别
            mask_img = np.argmax(pred_masks, axis=1)[0]
            # 对预测掩码进行后处理
            mask_img = self.postprocess(mask_img, transformer_info, (img_width, img_height))
            # 将处理后的掩码添加到列表中
            mask_images.append(mask_img)
        # 返回掩码图像列表
        return mask_images

    # 渲染图像函数
    def render_image(self, img, mask_img, output_path=None):
        # 使用可视化对象显示结果
        self.visualize.show_result(img, mask_img, output_path)

        # ade20k_metadata = MetadataCatalog.get("ade20k_sem_seg_val")
        # v = Visualizer(np.array(img), ade20k_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # semantic_result = v.draw_sem_seg(mask_img).get_image()
        # if output_path is not None:
        #     cv2.imwrite(output_path, semantic_result)
        # else:
        #     cv2.imshow(semantic_result)
        #     cv2.waitKey(0)


# 获取命令行参数函数
def get_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Mask2Former推理演示')
    # 添加配置文件路径参数
    parser.add_argument('--config', type=str, default='configs/maskformer_nuimages.yaml',
                        help='配置文件路径')
    # 添加输入图像路径或目录参数
    parser.add_argument('--input', type=str, default=r"C:\Users\19046\Desktop\detect_demo",
                        help='输入图像路径或目录')
    # 添加输出结果保存目录参数
    parser.add_argument('--output', type=str, default='output',
                        help='输出结果保存目录')
    # 添加GPU编号参数
    parser.add_argument('--local_rank', type=int, default=0,
                        help='GPU编号')
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
    # return parser.parse_args()


# 主函数
def main():
    # 获取命令行参数
    args = get_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载配置文件
    cfg = args
    # 设置本地rank
    cfg.local_rank = args.local_rank

    # 更新配置中的测试目录和保存目录
    cfg.TEST.TEST_DIR = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    cfg.TEST.SAVE_DIR = args.output

    # 创建分割推理器
    segmentor = Segmentation(cfg)

    # 处理单张图像或图像目录
    if os.path.isfile(args.input):
        # 处理单张图像
        image_paths = [args.input]
    else:
        # 处理目录中的所有图像
        image_paths = []
        # 遍历所有支持的图像扩展名
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            # 扩展图像路径列表
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))

    # 打印找到的图像数量
    print(f"找到 {len(image_paths)} 张图像进行推理")

    # 执行推理
    mask_images = segmentor.forward(image_paths)

    # 保存结果
    for i, (image_path, mask_img) in enumerate(zip(image_paths, mask_images)):
        # 读取原始图像
        original_img = Image.open(image_path).convert('RGB')

        # 生成输出文件路径
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        # 修改：确保输出路径使用PNG格式以支持RGBA模式
        output_path = os.path.join(args.output, f"{name}_seg.png")

        # 保存可视化结果
        segmentor.render_image(original_img, mask_img, output_path)

        # 保存分割掩码
        mask_output_path = os.path.join(args.output, f"{name}_mask.png")
        cv2.imwrite(mask_output_path, mask_img.astype(np.uint8))

        # 打印保存结果信息
        print(f"已保存结果: {output_path}")

    # 打印推理完成信息
    print("推理完成！")


# 程序入口点
if __name__ == "__main__":
    # 调用主函数
    main()