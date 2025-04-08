#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: evaluate_cblprd.py
@description: 评估CRNN模型在CBLPRD数据集上的性能

Usage:
    $ python evaluate_cblprd.py /path/to/CBLPRD-330k/ /path/to/model.pth --batch-size 64 --device 0
"""

import argparse
import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.model.crnn import CRNN
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.torchutil import select_device
from utils.logger import LOGGER
from utils.dataset.cblprd import CBLPRDDataset, PLATE_CHARS


def parse_opt():
    parser = argparse.ArgumentParser(description='CBLPRD 模型评估')
    parser.add_argument('data', type=str, help='CBLPRD数据集路径')
    parser.add_argument('model_path', type=str, help='模型权重文件路径')
    parser.add_argument('--test-txt', type=str, default='val.txt', 
                       help='测试文件列表路径。如果是相对路径，则相对于数据集根目录；如果是绝对路径，则直接使用该路径')
    parser.add_argument('--batch-size', type=int, default=64, help='批处理大小')
    
    parser.add_argument('--use-lstm', action='store_true', help='使用LSTM替代GRU')
    parser.add_argument('--not-tiny', action='store_true', help='使用非tiny版本的CRNN')
    
    # 数据处理选项
    parser.add_argument('--correct-skew', action='store_true', help='应用车牌倾斜校正')
    parser.add_argument('--process-double', action='store_true', help='处理双层车牌')
    
    parser.add_argument('--device', default='', help='CUDA设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--save-results', action='store_true', help='保存评估结果到文件')
    parser.add_argument('--output', type=str, default='./results', help='结果输出目录')
    
    args = parser.parse_args()
    return args


def evaluate(model, dataloader, criterion, device, blank_label=0):
    """评估模型性能"""
    model.eval()
    evaluator = Evaluator(blank_label=blank_label)
    total_loss = 0.0
    
    # 记录识别结果
    results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets_converted = dataloader.dataset.convert(targets)
            target_lengths = torch.IntTensor([len(t) for t in targets_converted]).to(device)
            targets_concat = torch.concat(targets_converted).to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets_concat, target_lengths)
            total_loss += loss.item()
            
            # 计算准确率
            acc = evaluator.update(outputs.cpu(), targets_converted)
            
            # 手动解码预测结果
            preds = outputs.cpu().detach().numpy()
            pred_texts = []
            for i in range(preds.shape[0]):
                pred = preds[i]
                pred_index = np.argmax(pred, axis=1)
                
                # 合并重复字符
                merged_chars = []
                prev_char = -1
                for p in pred_index:
                    if p != prev_char and p != blank_label:
                        merged_chars.append(p)
                    prev_char = p
                
                # 转换为字符串
                pred_text = ""
                for c in merged_chars:
                    if 0 <= c < len(PLATE_CHARS):
                        pred_text += PLATE_CHARS[c]
                
                pred_texts.append(pred_text)
            
            for i, (pred, gt) in enumerate(zip(pred_texts, targets)):
                match = pred == gt
                results.append((gt, pred, match))
            
            # 更新进度条
            avg_loss = total_loss / (idx + 1)
            current_acc = evaluator.result()
            pbar.set_description(f"Loss: {avg_loss:.6f}, Acc: {current_acc*100:.3f}%")
    
    # 计算最终指标
    final_acc = evaluator.result()
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': final_acc,
        'loss': avg_loss,
        'results': results
    }


def save_evaluation_results(results, model_path, output_dir):
    """保存评估结果到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 提取模型名称作为文件名前缀
    model_name = os.path.basename(model_path).split('.')[0]
    output_file = os.path.join(output_dir, f"{model_name}_evaluation.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {results['accuracy']*100:.4f}%\n")
        f.write(f"Loss: {results['loss']:.6f}\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Ground Truth':<20} | {'Prediction':<20} | {'Match':<5}\n")
        f.write("-" * 80 + "\n")
        
        for gt, pred, match in results['results']:
            match_str = "✓" if match else "✗"
            f.write(f"{gt:<20} | {pred:<20} | {match_str:<5}\n")
    
    LOGGER.info(f"保存评估结果到: {output_file}")
    return output_file


def main(opt):
    # 设置设备
    device = select_device(opt.device)
    
    # 设定输入形状 (W, H) - 适配CBLPRD-330k数据集的128×48尺寸
    input_shape = (128, 48)
    
    # 创建模型
    LOGGER.info("=> 创建模型")
    model = CRNN(in_channel=3, num_classes=len(PLATE_CHARS), cnn_input_height=input_shape[1],
                is_tiny=not opt.not_tiny, use_gru=not opt.use_lstm).to(device)
    
    # 加载模型权重
    LOGGER.info(f"=> 加载模型权重: {opt.model_path}")
    try:
        model.load_state_dict(torch.load(opt.model_path, map_location=device))
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
        if is_abs_path:
            # 如果是绝对路径，需要自定义数据加载过程
            import glob
            from utils.dataset.cblprd import load_txt_data, CBLPRDDataset
            
            # 从文件中读取图像路径和标签
            with open(test_txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 创建一个临时文件，用于存储调整后的路径
            temp_txt_path = os.path.join(os.path.dirname(test_txt_path), "temp_eval.txt")
            with open(temp_txt_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    f.write(line + "\n")
            
            # 使用临时文件加载数据集
            test_dataset = CBLPRDDataset(opt.data, temp_txt_path, is_train=False, 
                                      input_shape=input_shape,
                                      use_resampling=False, 
                                      correct_skew=opt.correct_skew,
                                      process_double=opt.process_double)
            
            # 删除临时文件
            os.remove(temp_txt_path)
        else:
            # 使用相对路径的常规方式
            test_dataset = CBLPRDDataset(opt.data, test_txt_path, is_train=False, 
                                      input_shape=input_shape,
                                      use_resampling=False, 
                                      correct_skew=opt.correct_skew,
                                      process_double=opt.process_double)
        
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                   num_workers=4, drop_last=False, pin_memory=True)
    except Exception as e:
        LOGGER.error(f"加载测试数据集失败: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return
    
    LOGGER.info(f"=> 开始评估 (测试集大小: {len(test_dataset)})")
    t0 = time.time()
    results = evaluate(model, test_dataloader, criterion, device, blank_label)
    eval_time = time.time() - t0
    
    # 输出评估结果
    LOGGER.info(f"\n评估结果:")
    LOGGER.info(f"准确率: {results['accuracy']*100:.4f}%")
    LOGGER.info(f"损失值: {results['loss']:.6f}")
    LOGGER.info(f"评估用时: {eval_time:.2f} 秒")
    
    # 计算错误率统计
    total = len(results['results'])
    correct = sum(1 for _, _, match in results['results'] if match)
    error_rate = (total - correct) / total * 100
    LOGGER.info(f"识别正确: {correct}/{total}, 错误率: {error_rate:.2f}%")
    
    # 保存评估结果
    if opt.save_results:
        output_file = save_evaluation_results(results, opt.model_path, opt.output)
        LOGGER.info(f"详细评估结果已保存到: {output_file}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
