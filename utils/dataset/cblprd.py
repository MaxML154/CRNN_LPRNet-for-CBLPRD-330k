# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: cblprd.py
@author: MaxML154
@description: Dataset handler for CBLPRD-330k dataset with support for various plate types
"""

import os
import random
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter

RANK = int(os.getenv('RANK', -1))

# 扩展车牌字符集，确保包含CBLPRD-330k中所有可能出现的字符
# 包括车牌中可能出现的后缀字符："学"、"临"、"挂"、"港"、"澳"、"使"、"领"等
PLATE_CHARS = "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领0123456789ABCDEFGHJKLMNPQRSTUVWXYZ临"

PLATE_DICT = dict()
for i in range(len(PLATE_CHARS)):
    PLATE_DICT[PLATE_CHARS[i]] = i


def is_plate_right(plate_name):
    """检查车牌是否只包含支持的字符"""
    assert isinstance(plate_name, str), plate_name
    for ch in plate_name:
        if ch not in PLATE_CHARS:
            return False
    return True


def correct_plate_skew(img, max_angle=15):
    """
    使用霍夫变换矫正车牌倾斜
    Args:
        img: 输入图像
        max_angle: 最大矫正角度(度)
    Returns:
        矫正后的图像
    """
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 边缘检测
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # 霍夫变换找线
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # 如果没有检测到线，返回原图
    if lines is None or len(lines) == 0:
        return img
    
    # 计算倾斜角度
    angles = []
    for line in lines:
        rho, theta = line[0]
        # 只关注接近水平的线(接近0度或180度)
        if (theta < np.pi/4 or theta > 3*np.pi/4):
            angle = theta * 180 / np.pi
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
    
    # 如果没有合适的线，返回原图
    if not angles:
        return img
    
    # 取中位数作为最终角度
    angle = np.median(angles)
    
    # 限制最大矫正角度
    if abs(angle) > max_angle:
        angle = max_angle if angle > 0 else -max_angle
    
    # 旋转图像矫正倾斜
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def process_double_layer_plate(img):
    """
    处理双层车牌，将双层车牌拼接成单层车牌
    Args:
        img: 输入双层车牌图像
    Returns:
        拼接后的单层车牌图像
    """
    h, w, c = img.shape
    # 上半部分车牌
    img_upper = img[0:int(5/12*h), :]
    # 下半部分车牌
    img_lower = img[int(1/3*h):, :]
    # 调整上半部分大小与下半部分匹配
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    # 水平拼接
    new_img = np.hstack((img_upper, img_lower))
    return new_img


def check_if_double_layer(img, plate_text):
    """
    检查是否为双层车牌
    Args:
        img: 输入图像
        plate_text: 车牌文本
    Returns:
        是否双层车牌的布尔值
    """
    # 判断依据：
    # 1. 通过车牌文本判断：拖拉机绿牌、黄牌挂车等
    if "挂" in plate_text:
        return True
    # 部分车牌特征
    # 湘18EZZG1 - 拖拉机绿牌
    if len(plate_text) >= 3 and plate_text[0] in "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新" and plate_text[1:3].isdigit():
        return True
    
    # 2. 通过高宽比来进一步确认
    h, w = img.shape[:2]
    aspect_ratio = w / h
    # 单层车牌宽高比通常大于2.5，双层通常小于2.5
    if aspect_ratio < 2.0:
        return True
    
    return False


def load_txt_data(txt_path, data_root):
    """从txt文件加载数据路径和标签"""
    assert os.path.isfile(txt_path), f"File not found: {txt_path}"
    
    data_list = []
    label_dict = {}
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 2:
                continue
                
            img_path = parts[0]  # 图像相对路径
            label_name = parts[1]  # 车牌号
            
            # 如果有第三个部分，它是车牌类型（如"单层黄牌"），现在保存起来供后续使用
            plate_type = parts[2] if len(parts) > 2 else ""
            
            # 验证车牌号是否合法
            if len(label_name) < 3:
                continue
            if not is_plate_right(label_name):
                continue
                
            # 构建完整图像路径
            full_img_path = os.path.join(data_root, img_path)
            if not os.path.isfile(full_img_path):
                print(f"Warning: Image file not found: {full_img_path}")
                continue
                
            # 存储车牌号及其对应的字符索引
            if label_name not in label_dict:
                label = []
                for i in range(len(label_name)):
                    label.append(PLATE_DICT[label_name[i]])
                label_dict[label_name] = label
                
            data_list.append([full_img_path, label_name, plate_type])
            
    return data_list, label_dict


class CBLPRDDataset(Dataset):
    """CBLPRD-330k数据集加载器，支持双层车牌处理和重采样"""

    def __init__(self, data_root, txt_file, is_train=True, input_shape=(128, 48),
                 use_resampling=True, correct_skew=True, process_double=True):
        """
        初始化CBLPRD-330k数据集
        
        Args:
            data_root: 数据集根目录
            txt_file: train.txt或val.txt文件路径（相对于data_root）
            is_train: 是否为训练模式
            input_shape: 模型输入形状 (宽, 高)，CBLPRD-330k数据集原始尺寸为(128, 48)
            use_resampling: 是否使用重采样减轻长尾效应
            correct_skew: 是否进行倾斜矫正
            process_double: 是否处理双层车牌
        """
        self.data_root = data_root
        self.is_train = is_train
        self.input_shape = input_shape
        self.correct_skew = correct_skew
        self.process_double = process_double
        
        # 构建txt文件的完整路径
        txt_path = os.path.join(data_root, txt_file)
        assert os.path.isfile(txt_path), f"Txt file not found: {txt_path}"
        
        # 加载数据
        data_list, label_dict = load_txt_data(txt_path, data_root)
        if RANK in {-1, 0}:
            print(f"Load {'train' if is_train else 'val'} data: {len(data_list)}")
            
        # 如果是训练模式并启用重采样，则进行重采样以平衡数据
        if is_train and use_resampling:
            data_list = self._resample_data(data_list)
            if RANK in {-1, 0}:
                print(f"After resampling: {len(data_list)} samples")
        
        self.data_list = data_list
        self.dataset_len = len(data_list)
        self.label_dict = label_dict
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10, fill=0),  # 较小的旋转角度
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 仿射变换
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色变换
        ])

    def _resample_data(self, data_list, min_samples=5, max_factor=10):
        """
        对不同类型的车牌进行重采样，减轻长尾效应
        
        Args:
            data_list: 原始数据列表
            min_samples: 每个类别的最小样本数
            max_factor: 重采样的最大倍数限制
        
        Returns:
            重采样后的数据列表
        """
        # 统计不同前缀车牌的数量
        province_counts = Counter()
        for item in data_list:
            _, plate_text, _ = item
            if len(plate_text) > 0:
                province = plate_text[0]  # 取第一个字符作为省份标识
                province_counts[province] += 1
        
        # 计算中位数样本数
        median_count = np.median(list(province_counts.values()))
        
        # 重采样数据
        resampled_data = []
        for item in data_list:
            _, plate_text, _ = item
            if len(plate_text) > 0:
                province = plate_text[0]
                count = province_counts[province]
                
                # 计算重采样权重：样本数少的类别权重高
                if count < median_count:
                    # 限制重采样倍数
                    weight = min(median_count / count, max_factor)
                    # 至少采样min_samples次
                    weight = max(weight, min_samples / count) if count < min_samples else weight
                    # 生成采样数
                    num_samples = math.ceil(weight)
                    # 添加重复样本
                    resampled_data.extend([item] * num_samples)
                else:
                    # 样本数多的类别保持原样
                    resampled_data.append(item)
            else:
                resampled_data.append(item)
        
        return resampled_data

    def __getitem__(self, index):
        assert index < self.dataset_len
        
        img_path, label_name, plate_type = self.data_list[index]
        image = cv2.imread(img_path)
        if image is None:
            # 如果图像加载失败，返回一个黑色图像
            print(f"Warning: Cannot load image {img_path}, using a black image instead")
            image = np.zeros((self.input_shape[1], self.input_shape[0], 3), dtype=np.uint8)
        
        if image.shape[-1] == 4:  # 处理RGBA图像
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # 倾斜矫正
        if self.correct_skew:
            image = correct_plate_skew(image)
        
        # 处理双层车牌
        if self.process_double and check_if_double_layer(image, label_name):
            image = process_double_layer_plate(image)
            
        # 数据增强
        if self.is_train and random.random() > 0.5:
            image_pil = self.transform(image)
            image = np.array(image_pil, dtype=np.uint8)
            
        # 调整图像大小
        if image.shape[0] != self.input_shape[1] or image.shape[1] != self.input_shape[0]:
            image = cv2.resize(image, self.input_shape)
        
        # 转换为Tensor
        data = torch.from_numpy(image).float() / 255.
        # HWC -> CHW
        data = data.permute(2, 0, 1)
        
        return data, label_name

    def __len__(self):
        return self.dataset_len
        
    def convert(self, targets):
        """将标签转换为模型可用的格式"""
        labels = []
        for label_name in targets:
            label = self.label_dict[label_name]
            labels.append(torch.IntTensor(label))
        return labels 
