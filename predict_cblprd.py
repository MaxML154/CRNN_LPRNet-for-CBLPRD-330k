# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: predict_cblprd.py
@author: [Your Name]
@description: Prediction script for CBLPRD-330k dataset

Usage - Predict using CRNN_Tiny/CRNN:
    $ python predict_cblprd.py crnn_tiny-cblprd.pth ./sample_image.jpg runs/predict/cblprd/
    $ python predict_cblprd.py crnn-cblprd.pth ./sample_image.jpg runs/predict/cblprd/ --not-tiny
"""

import os
import argparse
import time
from itertools import groupby

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的"-"负号的乱码问题

import importlib

# 根据脚本是否作为主模块运行来决定导入方式
if __name__ == '__main__':
    # 直接运行时，使用绝对导入
    PLATE_CHARS = importlib.import_module('utils.dataset.cblprd').PLATE_CHARS
    model_info = importlib.import_module('utils.general').model_info
    load_ocr_model = importlib.import_module('utils.general').load_ocr_model
    correct_plate_skew = importlib.import_module('utils.dataset.cblprd').correct_plate_skew
    process_double_layer_plate = importlib.import_module('utils.dataset.cblprd').process_double_layer_plate
    check_if_double_layer = importlib.import_module('utils.dataset.cblprd').check_if_double_layer
else:
    # 被导入时，尝试使用相对导入，如果失败则回退到绝对导入
    try:
        PLATE_CHARS = importlib.import_module('.utils.dataset.cblprd', package=__package__).PLATE_CHARS
        model_info = importlib.import_module('.utils.general', package=__package__).model_info
        load_ocr_model = importlib.import_module('.utils.general', package=__package__).load_ocr_model
        correct_plate_skew = importlib.import_module('.utils.dataset.cblprd', package=__package__).correct_plate_skew
        process_double_layer_plate = importlib.import_module('.utils.dataset.cblprd', package=__package__).process_double_layer_plate
        check_if_double_layer = importlib.import_module('.utils.dataset.cblprd', package=__package__).check_if_double_layer
    except ValueError:
        PLATE_CHARS = importlib.import_module('utils.dataset.cblprd').PLATE_CHARS
        model_info = importlib.import_module('utils.general').model_info
        load_ocr_model = importlib.import_module('utils.general').load_ocr_model
        correct_plate_skew = importlib.import_module('utils.dataset.cblprd').correct_plate_skew
        process_double_layer_plate = importlib.import_module('utils.dataset.cblprd').process_double_layer_plate
        check_if_double_layer = importlib.import_module('utils.dataset.cblprd').check_if_double_layer


def determine_plate_type(plate_text, original_image=None):
    """
    确定车牌类型
    Args:
        plate_text: 识别出的车牌文本
        original_image: 原始车牌图像，可以通过颜色进一步判断类型
    Returns:
        车牌类型描述
    """
    # 通过文字特征判断车牌类型
    if "挂" in plate_text:
        return "双层黄牌挂车"
    
    if "学" in plate_text:
        return "教练车牌"
    
    if "临" in plate_text:
        return "临时车牌"
    
    if "港" in plate_text:
        return "香港入出内地车牌"
    
    if "澳" in plate_text:
        return "澳门入出内地车牌"
    
    if "使" in plate_text:
        return "使馆车牌"
    
    if "领" in plate_text:
        return "领馆车牌"
    
    # 通过车牌长度和格式判断
    if len(plate_text) == 8:
        return "新能源车牌"
    
    # 通过前缀数字判断
    if len(plate_text) >= 3 and plate_text[0] in "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新" and plate_text[1:3].isdigit():
        return "拖拉机绿牌"
    
    # 默认为普通蓝牌
    return "普通蓝牌"


def parse_opt():
    parser = argparse.ArgumentParser(description='Predict CRNN with CBLPRD-330k')
    parser.add_argument('pretrained', type=str, help='path to pretrained model')
    parser.add_argument('image_path', type=str, help='path to image path')
    parser.add_argument('save_dir', type=str, help='path to save dir')
    
    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='Use this flag to specify non-tiny mode')
    
    # 添加车牌处理选项
    parser.add_argument('--correct-skew', action='store_true', help='apply skew correction to license plates')
    parser.add_argument('--process-double', action='store_true', help='process double-layer license plates')
    parser.add_argument('--no-skew-correction', action='store_true', help='disable skew correction')
    parser.add_argument('--no-double-process', action='store_true', help='disable double-layer processing')
    parser.add_argument('--determine-type', action='store_true', help='determine license plate type')
    
    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def predict_plate(image, model=None, device=None, img_h=48, img_w=128, correct_skew=False, process_double=False):
    start_time = time.time()
    
    # 保存原始图像用于车牌类型判断
    original_image = image.copy()
    
    # 倾斜矫正
    if correct_skew:
        image = correct_plate_skew(image)
    
    # 检查是否为双层车牌
    is_double_layer = False
    if process_double:
        is_double_layer = check_if_double_layer(image, '')
        if is_double_layer:
            image = process_double_layer_plate(image)
    
    # 调整输入图像大小以匹配CBLPRD-330k原始尺寸
    resize_image = cv2.resize(image, (img_w, img_h))
    
    data = torch.from_numpy(resize_image).float() / 255.
    # HWC -> CHW
    data = data.permute(2, 0, 1)
    
    # Infer
    data = data.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(data).cpu()[0]
        
    _, max_index = torch.max(output, dim=1)
    raw_pred = list(max_index.numpy())
    blank_label = 0
    pred = torch.IntTensor([c for c, _ in groupby(raw_pred) if c != blank_label])
    pred = pred.numpy()
    
    pred_plate = [PLATE_CHARS[i] for i in pred]
    plate_text = ''.join(pred_plate)
    
    # 判断车牌类型
    plate_type = determine_plate_type(plate_text, original_image)
    
    # 对于结果的处理，添加分隔符（在实际应用中，车牌显示格式可能根据车牌类型不同进行调整）
    if len(plate_text) >= 2:
        # 普通车牌：省份+城市代码后加点，如"粤A·12345"
        formatted_plate = plate_text[:2] + "·" + plate_text[2:]
    else:
        formatted_plate = plate_text
    
    end_time = time.time()
    predict_time = (end_time - start_time) * 1000
    
    # 返回更多信息
    result = {
        'plate_text': plate_text,  # 原始识别文本
        'formatted_plate': formatted_plate,  # 格式化后的显示
        'plate_type': plate_type,  # 车牌类型
        'is_double_layer': is_double_layer,  # 是否双层车牌
        'time_ms': predict_time  # 预测时间(毫秒)
    }
    
    print(f"Pred: {plate_text} ({plate_type}) - Predict time: {predict_time:.1f} ms")
    return result


def main():
    args = parse_opt()
    
    image_path = args.image_path
    assert os.path.isfile(image_path), image_path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    if image.shape[-1] == 4:  # 处理RGBA图像
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # 确定是否启用数据处理选项
    correct_skew = args.correct_skew and not args.no_skew_correction
    process_double = args.process_double and not args.no_double_process
    
    # 记录已启用的功能
    enabled_features = []
    if correct_skew:
        enabled_features.append("skew correction")
    if process_double:
        enabled_features.append("double-layer processing")
    
    if enabled_features:
        print(f"Enabled features: {', '.join(enabled_features)}")
    
    # 使用CBLPRD-330k数据集原始尺寸
    img_w = 128
    img_h = 48
    model, device = load_ocr_model(pretrained=args.pretrained, shape=(1, 3, img_h, img_w), 
                                  num_classes=len(PLATE_CHARS),
                                  not_tiny=args.not_tiny, use_lstm=args.use_lstm)
                                  
    # 预测
    result = predict_plate(image=image, model=model, device=device, img_h=img_h, img_w=img_w,
                         correct_skew=correct_skew, process_double=process_double)
    
    # 绘制预测结果
    plt.figure(figsize=(10, 4))
    
    # 构建标题
    if args.determine_type:
        title = f"预测结果: {result['formatted_plate']} ({result['plate_type']})"
    else:
        title = f"预测结果: {result['formatted_plate']}"
    
    plt.title(title, fontsize=14)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换BGR到RGB以正确显示颜色
    plt.axis('off')
    
    # 保存结果
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_name = os.path.basename(image_path)
    res_path = os.path.join(save_dir, f"plate_{image_name}")
    print(f'Save to {res_path}')
    plt.savefig(res_path)


if __name__ == '__main__':
    main() 