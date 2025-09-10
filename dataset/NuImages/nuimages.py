# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.
# nuScenes开发工具包
# 由Asha Asvathaman和Holger Caesar在2020年编写

import json
# 导入json模块，用于处理JSON格式数据
import os.path as osp
# 导入os.path模块并重命名为osp，用于处理文件路径
import sys
# 导入sys模块，用于访问Python解释器相关的变量
import time
# 导入time模块，用于处理时间相关功能
from collections import defaultdict
# 从collections模块导入defaultdict，用于创建带有默认值的字典
from typing import Any, List, Dict, Optional, Tuple, Callable
# 从typing模块导入类型提示相关类，用于函数参数和返回值的类型注解

import matplotlib.pyplot as plt
# 导入matplotlib.pyplot模块并重命名为plt，用于绘图功能
import numpy as np
# 导入numpy模块并重命名为np，用于数值计算
from PIL import Image, ImageDraw
# 从PIL模块导入Image和ImageDraw，用于图像处理和绘制
from pyquaternion import Quaternion
# 从pyquaternion模块导入Quaternion，用于四元数计算

from .utils import annotation_name, mask_decode, get_font, name_to_index_mapping
# 从当前包的utils模块导入annotation_name, mask_decode, get_font, name_to_index_mapping函数
from .color_map import get_colormap

# 从当前包的color_map模块导入get_colormap函数

PYTHON_VERSION = sys.version_info[0]
# 获取Python版本号，sys.version_info[0]表示主版本号

if not PYTHON_VERSION == 3:
    # 如果Python版本不是3
    raise ValueError("nuScenes dev-kit only supports Python version 3.")
    # 抛出值错误异常，提示nuScenes开发工具包只支持Python 3版本


class NuImages:
    # 定义NuImages类，用于处理nuImages数据集
    """
    Database class for nuImages to help query and retrieve information from the database.
    """

    # nuImages数据库类，用于帮助查询和检索数据库中的信息

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuimages',
                 lazy: bool = True,
                 verbose: bool = False):
        # 初始化函数，加载数据库并创建反向索引和快捷方式
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0-train", "v.0-val", "v1.0-test", "v1.0-mini").
        :param dataroot: Path to the tables and data.
        :param lazy: Whether to use lazy loading for the database tables.
        :param verbose: Whether to print status messages during load.
        """
        # 加载数据库并创建反向索引和快捷方式
        # :param version: 要加载的版本（例如"v1.0-train", "v1.0-val", "v1.0-test", "v1.0-mini"）
        # :param dataroot: 表格和数据的路径
        # :param lazy: 是否对数据库表使用延迟加载
        # :param verbose: 加载过程中是否打印状态消息

        self.version = version
        # 设置版本号
        self.dataroot = dataroot
        # 设置数据根目录
        self.lazy = lazy
        # 设置是否使用延迟加载
        self.verbose = verbose
        # 设置是否显示详细信息

        self.table_names = ['attribute', 'calibrated_sensor', 'category', 'ego_pose', 'log', 'object_ann', 'sample',
                            'sample_data', 'sensor', 'surface_ann']
        # 定义表名列表

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)
        # 断言数据库版本路径存在，如果不存在则抛出异常

        start_time = time.time()
        # 记录开始时间
        if verbose:
            # 如果显示详细信息
            print("======\nLoading nuImages tables for version {}...".format(self.version))
            # 打印加载信息

        # Init reverse indexing.
        # 初始化反向索引
        self._token2ind: Dict[str, Optional[dict]] = dict()
        # 创建_token2ind字典，用于存储token到索引的映射
        for table in self.table_names:
            # 遍历所有表名
            self._token2ind[table] = None
            # 初始化每个表的索引为None

        # Load tables directly if requested. 如果返回那么加载的tables字典
        # 如果请求直接加载表
        if not self.lazy:
            # 如果不使用延迟加载
            # Explicitly init tables to help the IDE determine valid class members.
            # 显式初始化表以帮助IDE确定有效的类成员
            self.attribute = self.__load_table__('attribute')
            # 加载attribute表
            self.calibrated_sensor = self.__load_table__('calibrated_sensor')
            # 加载calibrated_sensor表
            self.category = self.__load_table__('category')
            # 加载category表
            self.ego_pose = self.__load_table__('ego_pose')
            # 加载ego_pose表
            self.log = self.__load_table__('log')
            # 加载log表
            self.object_ann = self.__load_table__('object_ann')
            # 加载object_ann表
            self.sample = self.__load_table__('sample')
            # 加载sample表
            self.sample_data = self.__load_table__('sample_data')
            # 加载sample_data表
            self.sensor = self.__load_table__('sensor')
            # 加载sensor表
            self.surface_ann = self.__load_table__('surface_ann')
            # 加载surface_ann表

        self.color_map = get_colormap()  # 返回颜色
        # 获取颜色映射

        if verbose:
            # 如果显示详细信息
            print("Done loading in {:.3f} seconds (lazy={}).\n======".format(time.time() - start_time, self.lazy))
            # 打印加载完成信息，包括耗时和是否使用延迟加载

    # ### Internal methods. ###
    # 内部方法

    def __getattr__(self, attr_name: str) -> Any:
        # 获取属性的特殊方法，实现延迟加载数据库表
        """
        Implement lazy loading for the database tables. Otherwise throw the default error.
        :param attr_name: The name of the variable to look for.
        :return: The dictionary that represents that table.
        """
        # 为数据库表实现延迟加载，否则抛出默认错误
        # :param attr_name: 要查找的变量名
        # :return: 表示该表的字典

        if attr_name in self.table_names:
            # 如果属性名在表名列表中
            return self._load_lazy(attr_name, lambda tab_name: self.__load_table__(tab_name))
            # 延迟加载该表
        else:
            # 否则
            raise AttributeError("Error: %r object has no attribute %r" % (self.__class__.__name__, attr_name))
            # 抛出属性错误异常

    def get(self, table_name: str, token: str) -> dict:
        # 获取表中记录的方法
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        # 在常量时间内从表中返回一条记录
        # :param table_name: 表名
        # :param token: 记录的token
        # :return: 表记录，详见README.md中每个表的记录详情

        assert table_name in self.table_names, "Table {} not found".format(table_name)
        # 断言表名在表名列表中，否则抛出异常

        return getattr(self, table_name)[self.getind(table_name, token)]
        # 返回表中对应索引的记录

    def getind(self, table_name: str, token: str) -> int:
        # 获取记录在表中的索引
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        # 在常量时间内返回表中记录的索引
        # :param table_name: 表名
        # :param token: 记录的token
        # :return: 记录在表中的索引，表是一个数组

        # Lazy loading: Compute reverse indices.
        # 延迟加载：计算反向索引
        if self._token2ind[table_name] is None:
            # 如果该表的反向索引未计算
            self._token2ind[table_name] = dict()
            # 初始化该表的反向索引字典
            for ind, member in enumerate(getattr(self, table_name)):
                # 遍历表中的每个成员 todo 这里是按照顺序进行 为什么是反向索引呢
                self._token2ind[table_name][member['token']] = ind
                # 建立token到索引的映射
                # 如果是初始化那么建立一个table_name token(每个token不同) 的index 索引
        return self._token2ind[table_name][token]
        # 返回token对应的索引

    @property
    def table_root(self) -> str:
        # 表根目录属性
        """
        Returns the folder where the tables are stored for the relevant version.
        """
        # 返回相关版本的表存储文件夹

        return osp.join(self.dataroot, self.version)
        # 返回数据根目录和版本号拼接的路径

    def load_tables(self, table_names: List[str]) -> None:
        # 加载表的方法
        """
        Load tables and add them to self, if not already loaded.
        :param table_names: The names of the nuImages tables to be loaded.
        """
        # 加载表并添加到self中，如果尚未加载
        # :param table_names: 要加载的nuImages表名

        for table_name in table_names:
            # 遍历表名列表
            self._load_lazy(table_name, lambda tab_name: self.__load_table__(tab_name))
            # 延迟加载每个表

    def _load_lazy(self, attr_name: str, loading_func: Callable) -> Any:
        # 延迟加载属性的方法
        """
        Load an attribute and add it to self, if it isn't already loaded.
        :param attr_name: The name of the attribute to be loaded.
        :param loading_func: The function used to load it if necessary.
        :return: The loaded attribute.
        """
        # 加载属性并添加到self中，如果尚未加载
        # :param attr_name: 要加载的属性名
        # :param loading_func: 必要时用于加载的函数
        # :return: 加载的属性

        if attr_name in self.__dict__.keys():
            # 如果属性已加载
            return self.__getattribute__(attr_name)
            # 返回该属性
        else:
            # 否则
            attr = loading_func(attr_name)
            # 使用加载函数加载属性
            self.__setattr__(attr_name, attr)
            # 设置属性
            return attr
            # 返回属性

    def __load_table__(self, table_name) -> List[dict]:
        # 加载表的方法
        """
        Load a table and return it.
        :param table_name: The name of the table to load.
        :return: The table dictionary.
        """
        # 加载一个表并返回它
        # :param table_name: 要加载的表名
        # :return: 表字典

        start_time = time.time()
        # 记录开始时间
        table_path = osp.join(self.table_root, '{}.json'.format(table_name))
        # 构造表文件路径
        assert osp.exists(table_path), 'Error: Table %s does not exist!' % table_name
        # 断言表文件存在，否则抛出异常
        with open(table_path) as f:
            # 打开表文件
            table = json.load(f)
            # 加载JSON数据
        end_time = time.time()
        # 记录结束时间

        # Print a message to stdout.
        # 向标准输出打印消息
        if self.verbose:
            # 如果显示详细信息
            print("Loaded {} {}(s) in {:.3f}s,".format(len(table), table_name, end_time - start_time))
            # 打印加载信息

        return table
        # 返回表数据

    def shortcut(self, src_table: str, tgt_table: str, src_token: str) -> Dict[str, Any]:
        # 快捷方法，用于在不同表之间导航
        """
        Convenience function to navigate between different tables that have one-to-one relations.
        E.g. we can use this function to conveniently retrieve the sensor for a sample_data.
        :param src_table: The name of the source table.
        :param tgt_table: The name of the target table.
        :param src_token: The source token.
        :return: The entry of the destination table corresponding to the source token.
        """
        # 便利函数，用于在具有一对一关系的不同表之间导航
        # 例如，我们可以使用此函数方便地检索sample_data的传感器
        # :param src_table: 源表名
        # :param tgt_table: 目标表名
        # :param src_token: 源token
        # :return: 与源token对应的目标表条目

        if src_table == 'sample_data' and tgt_table == 'sensor':
            # 如果源表是sample_data且目标表是sensor
            sample_data = self.get('sample_data', src_token)
            # 获取sample_data记录
            calibrated_sensor = self.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            # 获取calibrated_sensor记录
            sensor = self.get('sensor', calibrated_sensor['sensor_token'])
            # 获取sensor记录

            return sensor
            # 返回sensor记录
        elif (src_table == 'object_ann' or src_table == 'surface_ann') and tgt_table == 'sample':
            # 如果源表是object_ann或surface_ann且目标表是sample
            src = self.get(src_table, src_token)
            # 获取源表记录
            sample_data = self.get('sample_data', src['sample_data_token'])
            # 获取sample_data记录
            sample = self.get('sample', sample_data['sample_token'])
            # 获取sample记录

            return sample
            # 返回sample记录
        else:
            # 否则
            raise Exception('Error: Shortcut from %s to %s not implemented!' % (src_table, tgt_table))
            # 抛出异常，该快捷方式未实现

    def check_sweeps(self, filename: str) -> None:
        # 检查sweeps文件夹是否已下载
        """
        Check that the sweeps folder was downloaded if required.
        :param filename: The filename of the sample_data.
        """
        # 检查是否需要时已下载sweeps文件夹
        # :param filename: sample_data的文件名

        assert filename.startswith('samples') or filename.startswith('sweeps'), \
            'Error: You passed an incorrect filename to check_sweeps(). Please use sample_data[''filename''].'
        # 断言文件名以samples或sweeps开头，否则抛出异常

        if 'sweeps' in filename:
            # 如果文件名包含sweeps
            sweeps_dir = osp.join(self.dataroot, 'sweeps')
            # 构造sweeps目录路径
            if not osp.isdir(sweeps_dir):
                # 如果sweeps目录不存在
                raise Exception('Error: You are missing the "%s" directory! The devkit generally works without this '
                                'directory, but you cannot call methods that use non-keyframe sample_datas.'
                                % sweeps_dir)
                # 抛出异常，缺少sweeps目录

    # ### List methods. ###
    # 列表方法

    def list_attributes(self, sort_by: str = 'freq') -> None:
        # 列出所有属性及每个属性的注释数量
        """
        List all attributes and the number of annotations with each attribute.
        :param sort_by: Sorting criteria, e.g. "name", "freq".
        """
        # 列出所有属性及每个属性的注释数量
        # :param sort_by: 排序标准，例如"name", "freq"

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['attribute', 'object_ann'])
            # 加载attribute和object_ann表

        # Count attributes.
        # 统计属性
        attribute_freqs = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计属性频率
        for object_ann in self.object_ann:
            # 遍历所有对象注释
            for attribute_token in object_ann['attribute_tokens']:
                # 遍历对象注释中的属性token
                attribute_freqs[attribute_token] += 1
                # 增加属性频率计数

        # Sort entries.
        # 排序条目
        if sort_by == 'name':
            # 如果按名称排序
            sort_order = [i for (i, _) in sorted(enumerate(self.attribute), key=lambda x: x[1]['name'])]
            # 根据属性名称排序
        elif sort_by == 'freq':
            # 如果按频率排序
            attribute_freqs_order = [attribute_freqs[c['token']] for c in self.attribute]
            # 获取属性频率顺序
            sort_order = [i for (i, _) in
                          sorted(enumerate(attribute_freqs_order), key=lambda x: x[1], reverse=True)]
            # 根据属性频率降序排序
        else:
            # 否则
            raise Exception('Error: Invalid sorting criterion %s!' % sort_by)
            # 抛出异常，排序标准无效

        # Print to stdout.
        # 打印到标准输出
        format_str = '{:11} {:24.24} {:48.48}'
        # 定义格式化字符串
        print()
        # 打印空行
        print(format_str.format('Annotations', 'Name', 'Description'))
        # 打印表头
        for s in sort_order:
            # 遍历排序后的索引
            attribute = self.attribute[s]
            # 获取属性
            print(format_str.format(
                attribute_freqs[attribute['token']], attribute['name'], attribute['description']))
            # 打印属性信息

    def list_cameras(self) -> None:
        # 列出所有相机及每个相机的样本数量
        """
        List all cameras and the number of samples for each.
        """
        # 列出所有相机及每个相机的样本数量

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['sample', 'sample_data', 'calibrated_sensor', 'sensor'])
            # 加载sample, sample_data, calibrated_sensor, sensor表

        # Count cameras.
        # 统计相机
        cs_freqs = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计校准传感器频率
        channel_freqs = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计通道频率
        for calibrated_sensor in self.calibrated_sensor:
            # 遍历所有校准传感器
            sensor = self.get('sensor', calibrated_sensor['sensor_token'])
            # 获取传感器
            cs_freqs[sensor['channel']] += 1
            # 增加校准传感器频率计数
        for sample_data in self.sample_data:
            # 遍历所有样本数据
            if sample_data['is_key_frame']:
                # 如果是关键帧# Only use keyframes (samples).
                sensor = self.shortcut('sample_data', 'sensor', sample_data['token'])
                # 获取传感器
                channel_freqs[sensor['channel']] += 1
                # 增加通道频率计数

        # Print to stdout.
        # 打印到标准输出
        format_str = '{:15} {:7} {:25}'
        # 定义格式化字符串
        print()
        # 打印空行
        print(format_str.format('Calibr. sensors', 'Samples', 'Channel'))
        # 打印表头
        for channel in cs_freqs.keys():
            # 遍历通道键
            cs_freq = cs_freqs[channel]
            # 获取校准传感器频率
            channel_freq = channel_freqs[channel]
            # 获取通道频率
            print(format_str.format(
                cs_freq, channel_freq, channel))
            # 打印相机信息

    def list_categories(self, sample_tokens: List[str] = None, sort_by: str = 'object_freq') -> None:
        # 列出所有类别及它们的对象注释和表面注释数量
        """
        List all categories and the number of object_anns and surface_anns for them.
        :param sample_tokens: A list of sample tokens for which category stats will be shown.
        :param sort_by: Sorting criteria, e.g. "name", "object_freq", "surface_freq".
        """
        # 列出所有类别及它们的对象注释和表面注释数量
        # :param sample_tokens: 将显示类别统计信息的样本token列表
        # :param sort_by: 排序标准，例如"name", "object_freq", "surface_freq"

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['sample', 'object_ann', 'surface_ann', 'category'])
            # 加载sample, object_ann, surface_ann, category表

        # Count object_anns and surface_anns.
        # 统计对象注释和表面注释
        object_freqs = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计对象频率
        surface_freqs = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计表面频率
        if sample_tokens is not None:
            # 如果提供了样本token列表
            sample_tokens = set(sample_tokens)
            # 转换为集合

        for object_ann in self.object_ann:
            # 遍历所有对象注释
            sample = self.shortcut('object_ann', 'sample', object_ann['token'])
            # 获取样本
            if sample_tokens is None or sample['token'] in sample_tokens:
                # 如果没有提供样本token列表或样本token在列表中
                object_freqs[object_ann['category_token']] += 1
                # 增加对象频率计数

        for surface_ann in self.surface_ann:
            # 遍历所有表面注释
            sample = self.shortcut('surface_ann', 'sample', surface_ann['token'])
            # 获取样本
            if sample_tokens is None or sample['token'] in sample_tokens:
                # 如果没有提供样本token列表或样本token在列表中
                surface_freqs[surface_ann['category_token']] += 1
                # 增加表面频率计数

        # Sort entries.
        # 排序条目
        if sort_by == 'name':
            # 如果按名称排序
            sort_order = [i for (i, _) in sorted(enumerate(self.category), key=lambda x: x[1]['name'])]
            # 根据类别名称排序
        elif sort_by == 'object_freq':
            # 如果按对象频率排序
            object_freqs_order = [object_freqs[c['token']] for c in self.category]
            # 获取对象频率顺序
            sort_order = [i for (i, _) in sorted(enumerate(object_freqs_order), key=lambda x: x[1], reverse=True)]
            # 根据对象频率降序排序
        elif sort_by == 'surface_freq':
            # 如果按表面频率排序
            surface_freqs_order = [surface_freqs[c['token']] for c in self.category]
            # 获取表面频率顺序
            sort_order = [i for (i, _) in sorted(enumerate(surface_freqs_order), key=lambda x: x[1], reverse=True)]
            # 根据表面频率降序排序
        else:
            # 否则
            raise Exception('Error: Invalid sorting criterion %s!' % sort_by)
            # 抛出异常，排序标准无效

        # Print to stdout.
        # 打印到标准输出
        format_str = '{:11} {:12} {:24.24} {:48.48}'
        # 定义格式化字符串
        print()
        # 打印空行
        print(format_str.format('Object_anns', 'Surface_anns', 'Name', 'Description'))
        # 打印表头
        for s in sort_order:
            # 遍历排序后的索引
            category = self.category[s]
            # 获取类别
            category_token = category['token']
            # 获取类别token
            object_freq = object_freqs[category_token]
            # 获取对象频率
            surface_freq = surface_freqs[category_token]
            # 获取表面频率

            # Skip empty categories.
            # 跳过空类别
            if object_freq == 0 and surface_freq == 0:
                # 如果对象频率和表面频率都为0
                continue
                # 跳过

            name = category['name']
            # 获取类别名称
            description = category['description']
            # 获取类别描述
            print(format_str.format(
                object_freq, surface_freq, name, description))
            # 打印类别信息

    def list_anns(self, sample_token: str, verbose: bool = True) -> Tuple[List[str], List[str]]:
        # 列出样本的所有注释
        """
        List all the annotations of a sample.
        :param sample_token: Sample token.
        :param verbose: Whether to print to stdout.
        :return: The object and surface annotation tokens in this sample.
        """
        # 列出样本的所有注释
        # :param sample_token: 样本token
        # :param verbose: 是否打印到标准输出
        # :return: 该样本中的对象和表面注释token

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['sample', 'object_ann', 'surface_ann', 'category'])
            # 加载sample, object_ann, surface_ann, category表

        sample = self.get('sample', sample_token)
        # 获取样本
        key_camera_token = sample['key_camera_token']
        # 获取关键相机token
        object_anns = [o for o in self.object_ann if o['sample_data_token'] == key_camera_token]
        # 获取对象注释列表
        surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == key_camera_token]
        # 获取表面注释列表

        if verbose:
            # 如果显示详细信息
            print('Printing object annotations:')
            # 打印对象注释信息
            for object_ann in object_anns:
                # 遍历对象注释
                category = self.get('category', object_ann['category_token'])
                # 获取类别
                attribute_names = [self.get('attribute', at)['name'] for at in object_ann['attribute_tokens']]
                # 获取属性名称列表
                print('{} {} {}'.format(object_ann['token'], category['name'], attribute_names))
                # 打印对象注释信息

            print('\nPrinting surface annotations:')
            # 打印表面注释信息
            for surface_ann in surface_anns:
                # 遍历表面注释
                category = self.get('category', surface_ann['category_token'])
                # 获取类别
                print(surface_ann['token'], category['name'])
                # 打印表面注释信息

        object_tokens = [o['token'] for o in object_anns]
        # 获取对象token列表
        surface_tokens = [s['token'] for s in surface_anns]
        # 获取表面token列表
        return object_tokens, surface_tokens
        # 返回对象token列表和表面token列表

    def list_logs(self) -> None:
        # 列出所有日志及每个日志的样本数量
        """
        List all logs and the number of samples per log.
        """
        # 列出所有日志及每个日志的样本数量

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['sample', 'log'])
            # 加载sample和log表

        # Count samples.
        # 统计样本
        sample_freqs = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计样本频率
        for sample in self.sample:
            # 遍历所有样本
            sample_freqs[sample['log_token']] += 1
            # 增加样本频率计数

        # Print to stdout.
        # 打印到标准输出
        format_str = '{:6} {:29} {:24}'
        # 定义格式化字符串
        print()
        # 打印空行
        print(format_str.format('Samples', 'Log', 'Location'))
        # 打印表头
        for log in self.log:
            # 遍历所有日志
            sample_freq = sample_freqs[log['token']]
            # 获取样本频率
            logfile = log['logfile']
            # 获取日志文件名
            location = log['location']
            # 获取位置
            print(format_str.format(
                sample_freq, logfile, location))
            # 打印日志信息

    def list_sample_content(self, sample_token: str) -> None:
        # 列出给定样本的sample_data
        """
        List the sample_datas for a given sample.
        :param sample_token: Sample token.
        """
        # 列出给定样本的sample_data
        # :param sample_token: 样本token

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['sample', 'sample_data'])
            # 加载sample和sample_data表

        # Print content for each modality.
        # 打印每种模态的内容
        sample = self.get('sample', sample_token)
        # 获取样本
        sample_data_tokens = self.get_sample_content(sample_token)
        # 获取样本内容
        timestamps = np.array([self.get('sample_data', sd_token)['timestamp'] for sd_token in sample_data_tokens])
        # 获取时间戳数组
        rel_times = (timestamps - sample['timestamp']) / 1e6
        # 计算相对时间

        print('\nListing sample content...')
        # 打印样本内容列表信息
        print('Rel. time\tSample_data token')
        # 打印表头
        for rel_time, sample_data_token in zip(rel_times, sample_data_tokens):
            # 遍历相对时间和sample_data token
            print('{:>9.1f}\t{}'.format(rel_time, sample_data_token))
            # 打印相对时间和sample_data token

    def list_sample_data_histogram(self) -> None:
        # 显示每个样本的sample_data数量直方图
        """
        Show a histogram of the number of sample_datas per sample.
        """
        # 显示每个样本的sample_data数量直方图

        # Preload data if in lazy load to avoid confusing outputs.
        # 如果是延迟加载则预加载数据以避免混淆输出
        if self.lazy:
            # 如果使用延迟加载
            self.load_tables(['sample_data'])
            # 加载sample_data表

        # Count sample_datas for each sample.
        # 统计每个样本的sample_data
        sample_counts = defaultdict(lambda: 0)
        # 创建默认值为0的字典用于统计样本计数
        for sample_data in self.sample_data:
            # 遍历所有样本数据
            sample_counts[sample_data['sample_token']] += 1
            # 增加样本计数

        # Compute histogram.
        # 计算直方图
        sample_counts_list = np.array(list(sample_counts.values()))
        # 获取样本计数列表
        bin_range = np.max(sample_counts_list) - np.min(sample_counts_list)
        # 计算范围
        if bin_range == 0:
            # 如果范围为0
            values = [len(sample_counts_list)]
            # 设置值
            freqs = [sample_counts_list[0]]
            # 设置频率
        else:
            # 否则
            values, bins = np.histogram(sample_counts_list, bin_range)
            # 计算直方图
            freqs = bins[1:]
            # 获取频率（需要使用bins的右侧） To get the frequency we need to use the right side of the bin.

        # Print statistics.
        # 打印统计信息
        print('\nListing sample_data frequencies..')
        # 打印sample_data频率列表信息
        print('# images\t# samples')
        # 打印表头
        for freq, val in zip(freqs, values):
            # 遍历频率和值
            print('{:>8d}\t{:d}'.format(int(freq), int(val)))
            # 打印频率和值

    # ### Getter methods. ###
    # 获取器方法

    def get_sample_content(self,
                           sample_token: str) -> List[str]:
        # 对于给定样本，按时间顺序返回所有sample_data
        """
        For a given sample, return all the sample_datas in chronological order.
        :param sample_token: Sample token.
        :return: A list of sample_data tokens sorted by their timestamp.
        """
        # 对于给定样本，按时间顺序返回所有sample_data
        # :param sample_token: 样本token
        # :return: 按时间戳排序的sample_data token列表

        sample = self.get('sample', sample_token)
        # 获取样本
        key_sd = self.get('sample_data', sample['key_camera_token'])
        # 获取关键sample_data

        # Go forward.
        # 向前遍历
        cur_sd = key_sd
        # 当前sample_data设为关键sample_data
        forward = []
        # 初始化向前列表
        while cur_sd['next'] != '':
            # 当当前sample_data的下一个不为空时
            cur_sd = self.get('sample_data', cur_sd['next'])
            # 获取下一个sample_data
            forward.append(cur_sd['token'])
            # 添加token到向前列表

        # Go backward.
        # 向后遍历
        cur_sd = key_sd
        # 当前sample_data设为关键sample_data
        backward = []
        # 初始化向后列表
        while cur_sd['prev'] != '':
            # 当当前sample_data的上一个不为空时
            cur_sd = self.get('sample_data', cur_sd['prev'])
            # 获取上一个sample_data
            backward.append(cur_sd['token'])
            # 添加token到向后列表

        # Combine.
        # 合并
        result = backward[::-1] + [key_sd['token']] + forward
        # 结果为向后列表的逆序加上关键sample_data的token再加上向前列表

        return result
        # 返回结果

    def get_ego_pose_data(self,
                          sample_token: str,
                          attribute_name: str = 'translation') -> Tuple[np.ndarray, np.ndarray]:
        # 返回与此样本关联的<=13个sample_data的自我姿态数据
        """
        Return the ego pose data of the <= 13 sample_datas associated with this sample.
        The method return translation, rotation, rotation_rate, acceleration and speed.
        :param sample_token: Sample token.
        :param attribute_name: The ego_pose field to extract, e.g. "translation", "acceleration" or "speed".
        :return: (
            timestamps: The timestamp of each ego_pose.
            attributes: A matrix with sample_datas x len(attribute) number of fields.
        )
        """
        # 返回与此样本关联的<=13个sample_data的自我姿态数据
        # 该方法返回translation, rotation, rotation_rate, acceleration和speed
        # :param sample_token: 样本token
        # :param attribute_name: 要提取的ego_pose字段，例如"translation", "acceleration"或"speed"
        # :return: (
        #     timestamps: 每个ego_pose的时间戳
        #     attributes: sample_datas x len(attribute)字段数的矩阵
        # )

        assert attribute_name in ['translation', 'rotation', 'rotation_rate', 'acceleration', 'speed'], \
            'Error: The attribute_name %s is not a valid option!' % attribute_name
        # 断言属性名在有效选项中，否则抛出异常

        if attribute_name == 'speed':
            # 如果属性名是speed
            attribute_len = 1
            # 属性长度为1
        elif attribute_name == 'rotation':
            # 如果属性名是rotation
            attribute_len = 4
            # 属性长度为4
        else:
            # 否则
            attribute_len = 3
            # 属性长度为3

        sd_tokens = self.get_sample_content(sample_token)
        # 获取样本内容
        attributes = np.zeros((len(sd_tokens), attribute_len))
        # 初始化属性数组
        timestamps = np.zeros((len(sd_tokens)))
        # 初始化时间戳数组
        for i, sd_token in enumerate(sd_tokens):
            # 遍历sample_data token
            # Get attribute.
            # 获取属性
            sample_data = self.get('sample_data', sd_token)
            # 获取sample_data
            ego_pose = self.get('ego_pose', sample_data['ego_pose_token'])
            # 获取ego_pose
            attribute = ego_pose[attribute_name]
            # 获取属性

            # Store results.
            # 存储结果
            attributes[i, :] = attribute
            # 存储属性
            timestamps[i] = ego_pose['timestamp']
            # 存储时间戳

        return timestamps, attributes
        # 返回时间戳和属性

    def get_trajectory(self,
                       sample_token: str,
                       rotation_yaw: float = 0.0,
                       center_key_pose: bool = True) -> Tuple[np.ndarray, int]:
        # 获取自我车辆的轨迹，并可选择旋转和居中
        """
        Get the trajectory of the ego vehicle and optionally rotate and center it.
        :param sample_token: Sample token.
        :param rotation_yaw: Rotation of the ego vehicle in the plot.
            Set to None to use lat/lon coordinates.
            Set to 0 to point in the driving direction at the time of the keyframe.
            Set to any other value to rotate relative to the driving direction (in radians).
        :param center_key_pose: Whether to center the trajectory on the key pose.
        :return: (
            translations: A matrix with sample_datas x 3 values of the translations at each timestamp.
            key_index: The index of the translations corresponding to the keyframe (usually 6).
        )
        """
        # 获取自我车辆的轨迹，并可选择旋转和居中
        # :param sample_token: 样本token
        # :param rotation_yaw: 图中自我车辆的旋转
        #     设置为None以使用纬度/经度坐标
        #     设置为0以在关键帧时指向行驶方向
        #     设置为任何其他值以相对于行驶方向旋转（以弧度为单位）
        # :param center_key_pose: 是否将轨迹居中在关键姿态上
        # :return: (
        #     translations: 每个时间戳平移的sample_datas x 3值矩阵
        #     key_index: 与关键帧对应的平移索引（通常为6）
        # )

        # Get trajectory data.
        # 获取轨迹数据
        timestamps, translations = self.get_ego_pose_data(sample_token)
        # 获取自我姿态数据

        # Find keyframe translation and rotation.
        # 查找关键帧平移和旋转
        sample = self.get('sample', sample_token)
        # 获取样本
        sample_data = self.get('sample_data', sample['key_camera_token'])
        # 获取sample_data
        ego_pose = self.get('ego_pose', sample_data['ego_pose_token'])
        # 获取ego_pose
        key_rotation = Quaternion(ego_pose['rotation'])
        # 获取关键旋转
        key_timestamp = ego_pose['timestamp']
        # 获取关键时间戳
        key_index = [i for i, t in enumerate(timestamps) if t == key_timestamp][0]
        # 获取关键索引

        # Rotate points such that the initial driving direction points upwards.
        # 旋转点使初始行驶方向向上
        if rotation_yaw is not None:
            # 如果旋转yaw不为空
            rotation = key_rotation.inverse * Quaternion(axis=[0, 0, 1], angle=np.pi / 2 - rotation_yaw)
            # 计算旋转
            translations = np.dot(rotation.rotation_matrix, translations.T).T
            # 应用旋转

        # Subtract origin to have lower numbers on the axes.
        # 减去原点以使轴上的数字更小
        if center_key_pose:
            # 如果居中关键姿态
            translations -= translations[key_index, :]
            # 减去关键姿态的平移

        return translations, key_index
        # 返回平移和关键索引

    def get_segmentation(self,
                         sd_token: str) -> Tuple[np.ndarray, np.ndarray]:
        # 生成两个大小为H x W的分割掩码numpy数组
        """
        Produces two segmentation masks as numpy arrays of size H x W each, where H and W are the height and width
        of the camera image respectively:
            - semantic mask: A mask in which each pixel is an integer value between 0 to C (inclusive),
                             where C is the number of categories in nuImages. Each integer corresponds to
                             the index of the class in the category.json.
            - instance mask: A mask in which each pixel is an integer value between 0 to N, where N is the
                             number of objects in a given camera sample_data. Each integer corresponds to
                             the order in which the object was drawn into the mask.
        :param sd_token: The token of the sample_data to be rendered.
        :return: Two 2D numpy arrays (one semantic mask <int32: H, W>, and one instance mask <int32: H, W>).
        """
        # 生成两个大小为H x W的分割掩码numpy数组，其中H和W分别是相机图像的高度和宽度：
        #     - 语义掩码：每个像素是0到C（包含）之间的整数值的掩码，
        #                其中C是nuImages中的类别数。每个整数对应于category.json中类的索引。
        #     - 实例掩码：每个像素是0到N之间的整数值的掩码，其中N是给定相机sample_data中的对象数。
        #                每个整数对应于对象被绘制到掩码中的顺序。
        # :param sd_token: 要渲染的sample_data的token
        # :return: 两个2D numpy数组（一个语义掩码<int32: H, W>，一个实例掩码<int32: H, W>）

        # Validate inputs.
        # 验证输入
        sample_data = self.get('sample_data', sd_token)  # 这里只需要添加 一个是类型 一个是token
        # 获取sample_data
        assert sample_data['is_key_frame'], 'Error: Cannot render annotations for non keyframes!'
        # 断言是关键帧，否则抛出异常

        name_to_index = name_to_index_mapping(self.category)
        # 获取名称到索引的映射

        # Get image data.
        # 获取图像数据
        self.check_sweeps(sample_data['filename'])
        # 检查sweeps
        im_path = osp.join(self.dataroot, sample_data['filename'])
        # 构造图像路径  todo 用opencv进行优化 将这个方法作为单独的函数
        im = Image.open(im_path)
        # 打开图像

        (width, height) = im.size
        # 获取图像尺寸 这里初始化只是填充一个图像
        semseg_mask = np.zeros((height, width)).astype('int32')
        # 初始化语义分割掩码 全0初始化
        instanceseg_mask = np.zeros((height, width)).astype('int32')
        # 初始化实例分割掩码

        # Load stuff / surface regions.
        # 加载内容 / 表面区域  加载对应data_token
        surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == sd_token]
        # 获取表面注释

        # Draw stuff / surface regions.
        # 绘制内容/表面区域
        for ann in surface_anns:
            # 遍历表面注释
            # Get color and mask.
            # 获取颜色和掩码
            category_token = ann['category_token']
            # 获取类别token
            category_name = self.get('category', category_token)['name']
            # 获取类别名称
            if ann['mask'] is None:
                # 如果掩码为空
                continue
                # 跳过
            mask = mask_decode(ann['mask'])  #
            # 解码掩码

            # Draw mask for semantic segmentation.
            # 绘制语义分割掩码 这里假定为每个掩码的值都是不重复 那么可以在一张图像上填充多个mask
            semseg_mask[mask == 1] = name_to_index[category_name]  # 填充上面的 semseg_mask
            # 设置语义分割掩码

        # Load object instances.
        # 加载对象实例 这里和上面一样
        object_anns = [o for o in self.object_ann if o['sample_data_token'] == sd_token]
        # 获取对象注释

        # Sort by token to ensure that objects always appear in the instance mask in the same order.
        # 按token排序以确保对象始终以相同顺序出现在实例掩码中
        object_anns = sorted(object_anns, key=lambda k: k['token'])
        # 排序对象注释

        # Draw object instances.
        # 绘制对象实例
        # The 0 index is reserved for background; thus, the instances should start from index 1.
        # 0索引保留给背景；因此，实例应从索引1开始
        for i, ann in enumerate(object_anns, start=1):
            # 遍历对象注释，从1开始编号
            # Get color, box, mask and name.
            # 获取颜色、框、掩码和名称
            category_token = ann['category_token']
            # 获取类别token
            category_name = self.get('category', category_token)['name']
            # 获取类别名称
            if ann['mask'] is None or category_name == 'vehicle.ego':
                # 如果掩码为空或类别名称为'vehicle.ego'
                continue
                # 跳过
            mask = mask_decode(ann['mask'])  # 这里的mask 是batch 64 格式
            # 解码掩码

            # Draw masks for semantic segmentation and instance segmentation.
            # 绘制语义分割和实例分割的掩码
            semseg_mask[mask == 1] = name_to_index[category_name]
            # 设置语义分割掩码
            instanceseg_mask[mask == 1] = i
            # 设置实例分割掩码
        # 一个是语义分割 一个是实例分割
        return semseg_mask, instanceseg_mask
        # 返回语义分割掩码和实例分割掩码

    # ### Rendering methods. ###
    # 渲染方法
    # 对网络预测的输出进行渲染
    def render_predict(self, img, mask: np.ndarray):
        # 渲染预测结果的方法
        """
        :param img: 原始图片
        :param mask: 网络输出，格式：[h, w]
        """
        # :param img: 原始图片
        # :param mask: 网络输出，格式：[h, w]

        groud_img = img.convert('RGBA')
        # 转换图像为RGBA模式
        draw = ImageDraw.Draw(groud_img, 'RGBA')
        # 创建绘图对象

        name_to_index = name_to_index_mapping(self.category)
        # 获取名称到索引的映射
        index_to_name = {name_to_index[key]: key for key in name_to_index.keys()}
        # 创建索引到名称的映射
        mask_id_list = list(np.unique(mask[mask > 0]))
        # 获取掩码ID列表
        for id in mask_id_list:
            # 遍历掩码ID列表
            category_name = index_to_name[id]
            # 获取类别名称
            color = self.color_map[category_name]
            # 获取颜色
            binary_mask = np.zeros(mask.shape, dtype=np.uint8)
            # 初始化二值掩码
            binary_mask[mask == id] = 1
            # 设置二值掩码
            draw.bitmap((0, 0), Image.fromarray(binary_mask * 128), fill=tuple(color + (128,)))
            # 绘制位图
        return groud_img
        # 返回渲染后的图像

    def render_image(self,
                     sd_token: str,
                     annotation_type: str = 'all',
                     with_category: bool = False,
                     with_attributes: bool = False,
                     object_tokens: List[str] = None,
                     surface_tokens: List[str] = None,
                     render_scale: float = 1.0,
                     box_line_width: int = -1,
                     font_size: int = None,
                     out_path: str = None) -> None:
        # 渲染图像（sample_data），可选择叠加注释
        """
        Renders an image (sample_data), optionally with annotations overlaid.
        :param sd_token: The token of the sample_data to be rendered.
        :param annotation_type: The types of annotations to draw on the image; there are four options:
            'all': Draw surfaces and objects, subject to any filtering done by object_tokens and surface_tokens.
            'surfaces': Draw only surfaces, subject to any filtering done by surface_tokens.
            'objects': Draw objects, subject to any filtering done by object_tokens.
            'none': Neither surfaces nor objects will be drawn.
        :param with_category: Whether to include the category name at the top of a box.
        :param with_attributes: Whether to include attributes in the label tags. Note that with_attributes=True
            will only work if with_category=True.
        :param object_tokens: List of object annotation tokens. If given, only these annotations are drawn.
        :param surface_tokens: List of surface annotation tokens. If given, only these annotations are drawn.
        :param render_scale: The scale at which the image will be rendered. Use 1.0 for the original image size.
        :param box_line_width: The box line width in pixels. The default is -1.
            If set to -1, box_line_width equals render_scale (rounded) to be larger in larger images.
        :param font_size: Size of the text in the rendered image. Use None for the default size.
        :param out_path: The path where we save the rendered image, or otherwise None.
            If a path is provided, the plot is not shown to the user.
        """
        # 渲染图像（sample_data），可选择叠加注释
        # :param sd_token: 要渲染的sample_data的token
        # :param annotation_type: 要在图像上绘制的注释类型；有四个选项：
        #     'all'：绘制表面和对象，受object_tokens和surface_tokens进行的任何过滤影响
        #     'surfaces'：仅绘制表面，受surface_tokens进行的任何过滤影响
        #     'objects'：绘制对象，受object_tokens进行的任何过滤影响
        #     'none'：既不绘制表面也不绘制对象
        # :param with_category: 是否在框的顶部包含类别名称
        # :param with_attributes: 是否在标签标记中包含属性。注意with_attributes=True
        #     仅在with_category=True时有效
        # :param object_tokens: 对象注释token列表。如果给出，仅绘制这些注释
        # :param surface_tokens: 表面注释token列表。如果给出，仅绘制这些注释
        # :param render_scale: 图像渲染的比例。使用1.0表示原始图像大小
        # :param box_line_width: 框线宽度（像素）。默认为-1
        #     如果设置为-1，box_line_width等于render_scale（四舍五入）以在较大图像中更大
        # :param font_size: 渲染图像中文本的大小。使用None表示默认大小
        # :param out_path: 保存渲染图像的路径，否则为None
        #     如果提供了路径，则不会向用户显示图表

        # Validate inputs.
        # 验证输入
        sample_data = self.get('sample_data', sd_token)
        # 获取sample_data
        if not sample_data['is_key_frame']:
            # 如果不是关键帧
            assert annotation_type == 'none', 'Error: Cannot render annotations for non keyframes!'
            # 断言注释类型为'none'，否则抛出异常
            assert not with_attributes, 'Error: Cannot render attributes for non keyframes!'
            # 断言不包含属性，否则抛出异常
        if with_attributes:
            # 如果包含属性
            assert with_category, 'In order to set with_attributes=True, with_category must be True.'
            # 断言必须包含类别
        assert type(box_line_width) == int, 'Error: box_line_width must be an integer!'
        # 断言框线宽度必须是整数
        if box_line_width == -1:
            # 如果框线宽度为-1
            box_line_width = int(round(render_scale))
            # 设置框线宽度为render_scale四舍五入后的整数

        # Get image data.
        # 获取图像数据
        self.check_sweeps(sample_data['filename'])
        # 检查sweeps
        im_path = osp.join(self.dataroot, sample_data['filename'])
        # 构造图像路径
        im = Image.open(im_path)
        # 打开图像

        # Initialize drawing.
        # 初始化绘图
        if with_category and font_size is not None:
            # 如果包含类别且字体大小不为空
            font = get_font(font_size=font_size)
            # 获取字体
        else:
            # 否则
            font = None
            # 字体为空
        im = im.convert('RGBA')
        # 转换图像为RGBA模式
        draw = ImageDraw.Draw(im, 'RGBA')
        # 创建绘图对象

        annotations_types = ['all', 'surfaces', 'objects', 'none']
        # 定义注释类型列表
        assert annotation_type in annotations_types, \
            'Error: {} is not a valid option for annotation_type. ' \
            'Only {} are allowed.'.format(annotation_type, annotations_types)
        # 断言注释类型在有效选项中，否则抛出异常
        if annotation_type is not 'none':
            # 如果注释类型不是'none'
            if annotation_type == 'all' or annotation_type == 'surfaces':
                # 如果注释类型是'all'或'surfaces'
                # Load stuff / surface regions.
                # 加载内容/表面区域
                surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == sd_token]
                # 获取表面注释
                if surface_tokens is not None:
                    # 如果提供了表面token
                    sd_surface_tokens = set([s['token'] for s in surface_anns if s['token']])
                    # 获取sample_data表面token集合
                    assert set(surface_tokens).issubset(sd_surface_tokens), \
                        'Error: The provided surface_tokens do not belong to the sd_token!'
                    # 断言提供的表面token属于sd_token，否则抛出异常
                    surface_anns = [o for o in surface_anns if o['token'] in surface_tokens]
                    # 过滤表面注释

                # Draw stuff / surface regions.
                # 绘制内容/表面区域
                for ann in surface_anns:
                    # 遍历表面注释
                    # Get color and mask.
                    # 获取颜色和掩码
                    category_token = ann['category_token']
                    # 获取类别token
                    category_name = self.get('category', category_token)['name']
                    # 获取类别名称
                    color = self.color_map[category_name]
                    # 获取颜色
                    if ann['mask'] is None:
                        # 如果掩码为空
                        continue
                        # 跳过
                    mask = mask_decode(ann['mask'])
                    # 解码掩码

                    # Draw mask. The label is obvious from the color.
                    # 绘制掩码。标签从颜色中显而易见
                    draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))
                    # 绘制位图

            if annotation_type == 'all' or annotation_type == 'objects':
                # 如果注释类型是'all'或'objects'
                # Load object instances.
                # 加载对象实例
                object_anns = [o for o in self.object_ann if o['sample_data_token'] == sd_token]
                # 获取对象注释
                if object_tokens is not None:
                    # 如果提供了对象token
                    sd_object_tokens = set([o['token'] for o in object_anns if o['token']])
                    # 获取sample_data对象token集合
                    assert set(object_tokens).issubset(sd_object_tokens), \
                        'Error: The provided object_tokens do not belong to the sd_token!'
                    # 断言提供的对象token属于sd_token，否则抛出异常
                    object_anns = [o for o in object_anns if o['token'] in object_tokens]
                    # 过滤对象注释

                # Draw object instances.
                # 绘制对象实例
                for ann in object_anns:
                    # 遍历对象注释
                    # Get color, box, mask and name.
                    # 获取颜色、框、掩码和名称
                    color = self.color_map[category_name]
                    # 获取颜色
                    bbox = ann['bbox']
                    # 获取边界框
                    attr_tokens = ann['attribute_tokens']
                    # 获取属性token
                    attributes = [self.get('attribute', at) for at in attr_tokens]
                    # 获取属性列表
                    name = annotation_name(attributes, category_name, with_attributes=with_attributes)
                    # 获取注释名称
                    if ann['mask'] is not None:
                        # 如果掩码不为空
                        mask = mask_decode(ann['mask'])
                        # 解码掩码

                        # Draw mask, rectangle and text.
                        # 绘制掩码、矩形和文本
                        draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))
                        # 绘制位图
                        draw.rectangle(bbox, outline=color, width=box_line_width)
                        # 绘制矩形
                        if with_category:
                            # 如果包含类别
                            draw.text((bbox[0], bbox[1]), name, font=font)
                            # 绘制文本

        # Plot the image.
        # 绘制图像
        (width, height) = im.size
        # 获取图像尺寸
        pix_to_inch = 100 / render_scale
        # 计算像素到英寸的转换
        figsize = (height / pix_to_inch, width / pix_to_inch)
        # 计算图像尺寸
        plt.figure(figsize=figsize)
        # 创建图像
        plt.axis('off')
        # 关闭坐标轴
        plt.imshow(im)
        # 显示图像

        # Save to disk.
        # 保存到磁盘
        if out_path is not None:
            # 如果提供了输出路径
            plt.savefig(out_path, bbox_inches='tight', dpi=2.295 * pix_to_inch, pad_inches=0)
            # 保存图像
            plt.close()
            # 关闭图像

    def render_trajectory(self,
                          sample_token: str,
                          rotation_yaw: float = 0.0,
                          center_key_pose: bool = True,
                          out_path: str = None) -> None:
        # 渲染围绕注释关键帧的轨迹图
        """
        Render a plot of the trajectory for the clip surrounding the annotated keyframe.
        A red cross indicates the starting point, a green dot the ego pose of the annotated keyframe.
        :param sample_token: Sample token.
        :param rotation_yaw: Rotation of the ego vehicle in the plot.
            Set to None to use lat/lon coordinates.
            Set to 0 to point in the driving direction at the time of the keyframe.
            Set to any other value to rotate relative to the driving direction (in radians).
        :param center_key_pose: Whether to center the trajectory on the key pose.
        :param out_path: Optional path to save the rendered figure to disk.
            If a path is provided, the plot is not shown to the user.
        """
        # 渲染围绕注释关键帧的轨迹图
        # 红色十字表示起点，绿色点表示注释关键帧的自我姿态
        # :param sample_token: 样本token
        # :param rotation_yaw: 图中自我车辆的旋转
        #     设置为None以使用纬度/经度坐标
        #     设置为0以在关键帧时指向行驶方向
        #     设置为任何其他值以相对于行驶方向旋转（以弧度为单位）
        # :param center_key_pose: 是否将轨迹居中在关键姿态上
        # :param out_path: 可选的保存渲染图像到磁盘的路径
        #     如果提供了路径，则不会向用户显示图表

        # Get the translations or poses.
        # 获取平移或姿态
        translations, key_index = self.get_trajectory(sample_token, rotation_yaw=rotation_yaw,
                                                      center_key_pose=center_key_pose)
        # 获取轨迹

        # Render translations.
        # 渲染平移
        plt.figure()
        # 创建图像
        plt.plot(translations[:, 0], translations[:, 1])
        # 绘制轨迹
        plt.plot(translations[key_index, 0], translations[key_index, 1], 'go', markersize=10)
        # 绘制关键图像（绿色点）
        plt.plot(translations[0, 0], translations[0, 1], 'rx', markersize=10)
        # 绘制起点（红色十字）
        max_dist = translations - translations[key_index, :]
        # 计算最大距离
        max_dist = np.ceil(np.max(np.abs(max_dist)) * 1.05)  # Leave some margin.
        # 计算最大距离（留一些边距）
        max_dist = np.maximum(10, max_dist)
        # 确保最大距离至少为10
        plt.xlim([translations[key_index, 0] - max_dist, translations[key_index, 0] + max_dist])
        # 设置x轴限制
        plt.ylim([translations[key_index, 1] - max_dist, translations[key_index, 1] + max_dist])
        # 设置y轴限制
        plt.xlabel('x in meters')
        # 设置x轴标签
        plt.ylabel('y in meters')
        # 设置y轴标签

        # Save to disk.
        # 保存到磁盘
        if out_path is not None:
            # 如果提供了输出路径
            plt.savefig(out_path, bbox_inches='tight', dpi=150, pad_inches=0)
            # 保存图像
            plt.close()
            # 关闭图像


if __name__ == '__main__':
    # 主函数
    import os
    # 导入os模块
    import matplotlib.pyplot as plt

    # 导入matplotlib.pyplot模块

    nuim = NuImages(dataroot='/home/dataset/nuImages/ImageData/nuimages-v1.0-mini/')
    # 创建NuImages对象
    sample_idx = 0
    # 设置样本索引
    sample = nuim.sample[sample_idx]
    # 获取样本
    sd_token = sample['key_camera_token']
    # 获取关键相机token
    sample_data = nuim.get('sample_data', sd_token)
    # 获取sample_data

    im_path = os.path.join(nuim.dataroot, sample_data['filename'])
    # 构造图像路径
    img = Image.open(im_path)
    # 打开图像

    semseg_mask, instanceseg_mask = nuim.get_segmentation(sd_token)
    # 获取分割掩码
    render_img = nuim.render_predict(img, semseg_mask)
    # 渲染图像
    render_img.save('/home/dataset/nuImages/ImageData/render_test.png')
    # 保存渲染图像