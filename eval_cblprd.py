# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: eval_cblprd.py
@author: [Your Name]
@description: Evaluation script for CBLPRD-330k dataset

Usage - Single-GPU eval:
    $ python3 eval_cblprd.py crnn_tiny-cblprd.pth /path/to/CBLPRD-330k/
    $ python3 eval_cblprd.py crnn-cblprd.pth /path/to/CBLPRD-330k/ --not-tiny
"""

import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.general import load_ocr_model
from utils.dataset.cblprd import CBLPRDDataset, PLATE_CHARS
from utils.evaluator import Evaluator


def parse_opt():
    parser = argparse.ArgumentParser(description='Eval CRNN with CBLPRD-330k')
    parser.add_argument('pretrained', type=str, help='path to pretrained model')
    parser.add_argument('val_root', type=str, help='path to CBLPRD-330k dataset')
    
    parser.add_argument('--val-txt', type=str, default='val.txt', help='path to val.txt relative to data root')
    
    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='use this flag to specify non-tiny mode')
    
    # 添加车牌处理选项
    parser.add_argument('--correct-skew', action='store_true', help='apply skew correction to license plates')
    parser.add_argument('--process-double', action='store_true', help='process double-layer license plates')
    parser.add_argument('--no-skew-correction', action='store_true', help='disable skew correction even if specified')
    parser.add_argument('--no-double-process', action='store_true', help='disable double-layer processing even if specified')
    
    args = parser.parse_args()
    print(f"args: {args}")
    return args


@torch.no_grad()
def val(args, val_root, pretrained):
    # 设置输入形状为 (W, H) - 适配CBLPRD-330k原始尺寸
    img_w = 128
    img_h = 48
    model, device = load_ocr_model(pretrained=pretrained, shape=(1, 3, img_h, img_w), 
                                  num_classes=len(PLATE_CHARS),
                                  not_tiny=args.not_tiny, use_lstm=args.use_lstm)
    
    # 确定是否启用各数据处理选项
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
                                  
    val_dataset = CBLPRDDataset(val_root, args.val_txt, is_train=False, input_shape=(img_w, img_h),
                              use_resampling=False, correct_skew=correct_skew, process_double=process_double)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, 
                               drop_last=False, pin_memory=True)
    
    blank_label = 0
    evaluator = Evaluator(blank_label=blank_label)
    
    pbar = tqdm(val_dataloader)
    for idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = val_dataset.convert(targets)
        with torch.no_grad():
            outputs = model(images).cpu()
            
        acc = evaluator.update(outputs, targets)
        info = f"Batch:{idx} ACC:{acc * 100:.3f}"
        pbar.set_description(info)
    acc = evaluator.result()
    print(f"ACC:{acc * 100:.3f}")


def main():
    args = parse_opt()
    val(args, args.val_root, args.pretrained)


if __name__ == '__main__':
    main() 