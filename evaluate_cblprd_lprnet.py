#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: evaluate_cblprd_lprnet.py
@description: 评估LPRNet模型在CBLPRD数据集上的性能

Usage:
    $ python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ /path/to/model.pth --batch-size 64 --device 0

    使用LPRNet评估:
    $ python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ /path/to/lprnet-cblprd-b512-best.pth --use-origin-block --batch-size 64 --device 0

    使用LPRNetPlus评估:
    $ python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ /path/to/lprnet_plus-cblprd-b512-best.pth --batch-size 64 --device 0

    使用LPRNet+STNet评估:
    $ python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ /path/to/lprnet_stnet-cblprd-b512-best.pth --use-origin-block --add-stnet --batch-size 64 --device 0
"""

import argparse
import os
import time
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.model.lprnet import LPRNet
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.torchutil import select_device
from utils.logger import LOGGER
from utils.dataset.cblprd import CBLPRDDataset, PLATE_CHARS


def parse_opt():
    parser = argparse.ArgumentParser(description='CBLPRD LPRNet模型评估')
    parser.add_argument('data', type=str, help='CBLPRD数据集路径')
    parser.add_argument('model_path', type=str, help='模型权重文件路径')
    parser.add_argument('--test-txt', type=str, default='val.txt', 
                       help='测试文件列表路径。如果是相对路径，则相对于数据集根目录；如果是绝对路径，则直接使用该路径')
    parser.add_argument('--batch-size', type=int, default=64, help='批处理大小')
    
    # LPRNet模型选项
    parser.add_argument('--use-origin-block', action='store_true', help='使用原始的LPRNet实现')
    parser.add_argument('--add-stnet', action='store_true', help='添加STNet进行空间变换')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='dropout比率')
    
    # 数据处理选项
    parser.add_argument('--correct-skew', action='store_true', help='应用车牌倾斜校正')
    parser.add_argument('--no-double-process', action='store_true', help='禁用双层车牌处理（默认启用）')
    
    parser.add_argument('--device', default='', help='CUDA设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--save-results', action='store_true', help='保存评估结果到文件')
    parser.add_argument('--output', type=str, default='./results', help='结果输出目录')
    
    args = parser.parse_args()
    return args


def evaluate_model(model, val_loader, val_dataset, criterion, device, blank_label=0):
    """评估模型性能"""
    model.eval()
    evaluator = Evaluator(blank_label=blank_label)
    total_loss = 0
    
    # 用于保存识别结果
    predictions = []
    targets_list = []
    correct_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # 准备数据
            targets_tensors = val_dataset.convert(targets)
            target_lengths = torch.IntTensor([len(t) for t in targets_tensors]).to(device)
            targets_concat = torch.concat(targets_tensors).to(device)
            
            # 前向传播
            outputs = model(images.to(device))
            
            # 计算损失
            loss = criterion(outputs, targets_concat, target_lengths)
            total_loss += loss.item()
            
            # 更新评估指标
            acc = evaluator.update(outputs.cpu(), targets_tensors)
            
            # 获取预测结果
            batch_preds = []
            batch_targets = []
            for i, output in enumerate(outputs):
                pred = evaluator.decode(output.unsqueeze(0))
                batch_preds.append(pred[0])  # 只取第一个结果
                batch_targets.append(targets[i])
                
                # 检查预测是否正确
                if pred[0] == targets[i]:
                    correct_count += 1
                
            predictions.extend(batch_preds)
            targets_list.extend(batch_targets)
    
    # 计算指标
    total_time = time.time() - start_time
    total_samples = len(val_dataset)
    avg_loss = total_loss / len(val_loader)
    accuracy = evaluator.result()
    accuracy_per_sample = correct_count / total_samples
    
    results = {
        "test_loss": avg_loss,
        "accuracy": accuracy,
        "accuracy_per_sample": accuracy_per_sample,
        "samples": total_samples,
        "correct_samples": correct_count,
        "inference_time": total_time,
        "samples_per_second": total_samples / total_time
    }
    
    return results, predictions, targets_list


def analyze_errors(predictions, targets, output_path=None):
    """分析错误预测"""
    errors = []
    total = len(predictions)
    error_count = 0
    
    # 分析错误类型
    char_errors = {"insertion": 0, "deletion": 0, "substitution": 0}
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        if pred != target:
            error_count += 1
            # 简单比较长度来判断错误类型
            if len(pred) > len(target):
                char_errors["insertion"] += 1
            elif len(pred) < len(target):
                char_errors["deletion"] += 1
            else:
                char_errors["substitution"] += 1
                
            errors.append({
                "index": i,
                "prediction": pred,
                "target": target
            })
    
    error_rate = error_count / total * 100
    
    error_analysis = {
        "total_samples": total,
        "error_count": error_count,
        "error_rate": error_rate,
        "char_errors": char_errors,
        "errors": errors[:50]  # 只保存前50个错误样本，避免结果过大
    }
    
    # 保存分析结果到文件
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    return error_analysis


def main(opt):
    # 设置设备
    device = select_device(opt.device)
    
    # 设定输入形状 (W, H) - LPRNet使用较小的输入尺寸
    input_shape = (94, 24)
    
    # 创建模型
    LOGGER.info("=> 创建模型")
    model = LPRNet(in_channel=3, num_classes=len(PLATE_CHARS),
                  dropout_rate=opt.dropout_rate,
                  use_origin_block=opt.use_origin_block,
                  add_stnet=opt.add_stnet).to(device)
    
    # 确定模型类型
    if opt.use_origin_block:
        model_type = "LPRNet"
    else:
        model_type = "LPRNetPlus"
    if opt.add_stnet:
        model_type += "+STNet"
    
    LOGGER.info(f"=> 模型类型: {model_type}")
    
    # 加载模型权重
    LOGGER.info(f"=> 加载模型权重: {opt.model_path}")
    try:
        state_dict = torch.load(opt.model_path, map_location=device)
        # 处理可能的'module.'前缀（分布式训练产生的）
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        LOGGER.info("模型加载成功")
    except Exception as e:
        LOGGER.error(f"加载模型失败: {e}")
        return
    
    # 创建损失函数
    blank_label = 0
    criterion = CTCLoss(blank_label=blank_label).to(device)
    
    # 处理测试文件路径
    test_txt_path = opt.test_txt
    is_abs_path = os.path.isabs(test_txt_path)
    
    if is_abs_path:
        LOGGER.info(f"=> 使用绝对路径测试文件: {test_txt_path}")
    else:
        # 如果是相对路径，则相对于数据集根目录
        test_txt_path = os.path.join(opt.data, test_txt_path)
        LOGGER.info(f"=> 使用相对路径测试文件: {test_txt_path}")
    
    # 确保测试文件存在
    if not os.path.exists(test_txt_path):
        LOGGER.error(f"测试文件不存在: {test_txt_path}")
        return
    
    # 加载测试数据集
    LOGGER.info("=> 加载测试数据集")
    try:
        # 默认启用双层车牌处理，除非使用--no-double-process参数禁用
        process_double = not opt.no_double_process
        
        test_dataset = CBLPRDDataset(opt.data, test_txt_path, is_train=False, 
                                     input_shape=input_shape,
                                     use_resampling=False, 
                                     correct_skew=opt.correct_skew,
                                     process_double=process_double)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 记录数据处理选项
        enabled_features = []
        if opt.correct_skew:
            enabled_features.append("倾斜校正")
        if process_double:
            enabled_features.append("双层车牌处理")
            
        if enabled_features:
            LOGGER.info(f"已启用的数据处理功能: {', '.join(enabled_features)}")
            
        LOGGER.info(f"加载测试数据集成功, 共{len(test_dataset)}张图像")
    except Exception as e:
        LOGGER.error(f"加载测试数据集失败: {e}")
        return
    
    # 评估模型
    LOGGER.info("=> 开始评估模型")
    results, predictions, targets = evaluate_model(model, test_loader, test_dataset, criterion, device)
    
    # 打印评估结果
    LOGGER.info("\n===== 模型评估结果 =====")
    LOGGER.info(f"模型类型: {model_type}")
    LOGGER.info(f"测试集样本数: {results['samples']}")
    LOGGER.info(f"正确识别样本数: {results['correct_samples']}")
    LOGGER.info(f"测试集损失: {results['test_loss']:.6f}")
    LOGGER.info(f"字符准确率: {results['accuracy']*100:.2f}%")
    LOGGER.info(f"整体样本准确率: {results['accuracy_per_sample']*100:.2f}%")
    LOGGER.info(f"推理时间: {results['inference_time']:.2f}秒")
    LOGGER.info(f"每秒处理样本数: {results['samples_per_second']:.2f}")
    
    # 如果需要保存结果
    if opt.save_results:
        # 创建输出目录
        if not os.path.exists(opt.output):
            os.makedirs(opt.output)
        
        # 准备保存文件路径
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name = os.path.splitext(os.path.basename(opt.model_path))[0]
        results_path = os.path.join(opt.output, f"{model_name}_{timestamp}_results.json")
        errors_path = os.path.join(opt.output, f"{model_name}_{timestamp}_errors.json")
        
        # 保存总体结果
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        LOGGER.info(f"总体结果已保存至: {results_path}")
        
        # 分析并保存错误信息
        LOGGER.info("=> 分析错误预测")
        error_analysis = analyze_errors(predictions, targets, errors_path)
        LOGGER.info(f"错误率: {error_analysis['error_rate']:.2f}%")
        LOGGER.info(f"错误分析已保存至: {errors_path}")


if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 