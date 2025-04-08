#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: split_dataset.py
@description: 划分CBLPRD-330k数据集为训练集、验证集和测试集，保持各类别均衡分布

Usage:
    $ python split_dataset.py /path/to/data.txt --data-root /path/to/CBLPRD-330k --output-dir /path/to/output --ratio 7:2:1 --by-plate-type
"""

import os
import argparse
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def parse_opt():
    parser = argparse.ArgumentParser(description='CBLPRD-330k数据集划分工具')
    parser.add_argument('data_txt', type=str, help='数据集索引文件data.txt的路径')
    parser.add_argument('--data-root', type=str, default='', help='数据集根目录路径，如果不提供则使用data.txt所在目录')
    parser.add_argument('--output-dir', type=str, default='', help='输出文件保存目录，如果不提供则使用data.txt所在目录')
    parser.add_argument('--train-file', type=str, default='train.txt', help='训练集文件名')
    parser.add_argument('--val-file', type=str, default='val.txt', help='验证集文件名')
    parser.add_argument('--test-file', type=str, default='test.txt', help='测试集文件名')
    parser.add_argument('--ratio', type=str, default='7:2:1', help='训练集:验证集:测试集的比例，如7:2:1')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子，用于数据集划分的可重复性')
    parser.add_argument('--balance', action='store_true', help='是否均衡各类别的分布')
    parser.add_argument('--by-plate-type', action='store_true', help='是否按照车牌类型划分数据集')
    parser.add_argument('--output-by-type', action='store_true', help='是否为每种车牌类型生成单独的数据集文件')
    
    args = parser.parse_args()
    return args


def read_data_file(data_txt_path):
    """
    读取数据集索引文件
    Args:
        data_txt_path: 数据集索引文件路径
    Returns:
        数据列表，每项包含图像路径和标签
    """
    print(f"读取数据集索引文件: {data_txt_path}")
    data_list = []
    
    with open(data_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析每一行
    for line in tqdm(lines, desc="解析数据集索引"):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        img_path = parts[0]  # 图像相对路径
        label = parts[1]     # 车牌号
        
        # 添加车牌类型信息（如果有）
        plate_type = parts[2] if len(parts) > 2 else "未知类型"
        
        data_list.append((img_path, label, plate_type))
    
    print(f"共读取{len(data_list)}条数据")
    return data_list


def group_by_label(data_list):
    """
    按车牌标签分组
    Args:
        data_list: 数据列表
    Returns:
        按车牌标签分组的字典
    """
    label_groups = defaultdict(list)
    first_char_groups = defaultdict(list)
    
    for item in data_list:
        img_path, label, plate_type = item
        label_groups[label].append(item)
        
        # 也按照车牌的第一个字符（省份）分组
        if label and len(label) > 0:
            first_char = label[0]
            first_char_groups[first_char].append(item)
    
    return label_groups, first_char_groups


def group_by_plate_type(data_list):
    """
    按车牌类型分组
    Args:
        data_list: 数据列表
    Returns:
        按车牌类型分组的字典
    """
    plate_type_groups = defaultdict(list)
    
    for item in data_list:
        img_path, label, plate_type = item
        plate_type_groups[plate_type].append(item)
    
    return plate_type_groups


def analyze_dataset(data_list):
    """分析数据集分布情况"""
    label_groups, first_char_groups = group_by_label(data_list)
    plate_type_groups = group_by_plate_type(data_list)
    
    # 计算每个标签的样本数量
    label_counts = {label: len(items) for label, items in label_groups.items()}
    
    # 计算每个省份的样本数量
    province_counts = {province: len(items) for province, items in first_char_groups.items()}
    
    # 计算每个车牌类型的样本数量
    plate_type_counts = {plate_type: len(items) for plate_type, items in plate_type_groups.items()}
    
    # 按省份/第一个字符分布统计
    print("\n按省份分布统计:")
    provinces = sorted(province_counts.items(), key=lambda x: x[1], reverse=True)
    for province, count in provinces:
        print(f"{province}: {count}样本 ({count/len(data_list)*100:.2f}%)")
    
    # 按车牌类型分布统计
    print("\n按车牌类型分布统计:")
    plate_types = sorted(plate_type_counts.items(), key=lambda x: x[1], reverse=True)
    for plate_type, count in plate_types:
        print(f"{plate_type}: {count}样本 ({count/len(data_list)*100:.2f}%)")
    
    # 样本数量分布统计
    plate_count_stats = {}
    for count in label_counts.values():
        if count not in plate_count_stats:
            plate_count_stats[count] = 0
        plate_count_stats[count] += 1
    
    print("\n车牌出现次数统计:")
    for count, frequency in sorted(plate_count_stats.items()):
        print(f"出现{count}次的车牌有{frequency}个")
    
    # 计算均值和中位数
    counts = list(label_counts.values())
    mean_count = sum(counts) / len(counts)
    median_count = sorted(counts)[len(counts)//2]
    
    print(f"\n总共有{len(label_groups)}个不同的车牌")
    print(f"平均每个车牌出现{mean_count:.2f}次")
    print(f"中位数: {median_count}次")
    print(f"最少出现次数: {min(counts)}次")
    print(f"最多出现次数: {max(counts)}次")
    
    return label_groups, first_char_groups, plate_type_groups


def split_dataset(data_list, ratio=(7, 2, 1), balance=True, by_plate_type=False, random_seed=42):
    """
    划分数据集为训练集、验证集和测试集
    Args:
        data_list: 数据列表
        ratio: 训练集:验证集:测试集的比例
        balance: 是否均衡各类别的分布
        by_plate_type: 是否按照车牌类型划分
        random_seed: 随机种子
    Returns:
        训练集、验证集和测试集
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 计算比例总和
    total_ratio = sum(ratio)
    train_ratio, val_ratio, test_ratio = ratio
    
    train_data, val_data, test_data = [], [], []
    
    if balance:
        if by_plate_type:
            # 按车牌类型分组
            _, _, plate_type_groups = analyze_dataset(data_list)
            print(f"按车牌类型分组后共有{len(plate_type_groups)}个组")
            
            # 对每个车牌类型组分别划分
            for plate_type, items in tqdm(plate_type_groups.items(), desc="按车牌类型划分数据集"):
                # 打乱每个组内的顺序
                random.shuffle(items)
                
                # 计算每个集合应包含的样本数
                n_samples = len(items)
                n_train = max(1, int(n_samples * train_ratio / total_ratio))
                n_val = max(1, int(n_samples * val_ratio / total_ratio))
                
                # 确保所有样本都被分配
                n_test = n_samples - n_train - n_val
                
                # 特殊情况处理：样本太少时保证至少有1个测试样本
                if n_test <= 0 and n_samples > 2:
                    n_train = max(1, n_train - 1)
                    n_test = n_samples - n_train - n_val
                
                # 划分数据集
                train_data.extend(items[:n_train])
                val_data.extend(items[n_train:n_train + n_val])
                
                # 添加测试集（如果有足够样本）
                if n_test > 0:
                    test_data.extend(items[n_train + n_val:])
                
                print(f"车牌类型 {plate_type}: 总计{n_samples}样本，训练集{n_train}，验证集{n_val}，测试集{n_test}")
        else:
            # 按车牌标签分组
            label_groups, _ = group_by_label(data_list)
            print(f"按车牌标签分组后共有{len(label_groups)}个组")
            
            # 对每个组分别划分
            for label, items in tqdm(label_groups.items(), desc="划分数据集"):
                # 打乱每个组内的顺序
                random.shuffle(items)
                
                # 计算每个集合应包含的样本数
                n_samples = len(items)
                n_train = max(1, int(n_samples * train_ratio / total_ratio))
                n_val = max(1, int(n_samples * val_ratio / total_ratio))
                
                # 确保所有样本都被分配
                n_test = n_samples - n_train - n_val
                
                # 特殊情况处理：样本太少时保证至少有1个测试样本
                if n_test <= 0 and n_samples > 2:
                    n_train = max(1, n_train - 1)
                    n_test = n_samples - n_train - n_val
                
                # 划分数据集
                train_data.extend(items[:n_train])
                val_data.extend(items[n_train:n_train + n_val])
                
                # 添加测试集（如果有足够样本）
                if n_test > 0:
                    test_data.extend(items[n_train + n_val:])
    else:
        # 不需要均衡各类别，直接按比例划分全部数据
        random.shuffle(data_list)
        n_samples = len(data_list)
        n_train = int(n_samples * train_ratio / total_ratio)
        n_val = int(n_samples * val_ratio / total_ratio)
        
        train_data = data_list[:n_train]
        val_data = data_list[n_train:n_train + n_val]
        test_data = data_list[n_train + n_val:]
    
    print(f"\n数据集划分完成:")
    print(f"训练集: {len(train_data)}样本 ({len(train_data)/len(data_list)*100:.2f}%)")
    print(f"验证集: {len(val_data)}样本 ({len(val_data)/len(data_list)*100:.2f}%)")
    print(f"测试集: {len(test_data)}样本 ({len(test_data)/len(data_list)*100:.2f}%)")
    
    return train_data, val_data, test_data


def save_split_files(train_data, val_data, test_data, output_dir, train_file, val_file, test_file, output_by_type=False):
    """
    保存划分后的数据集
    Args:
        train_data: 训练集数据
        val_data: 验证集数据
        test_data: 测试集数据
        output_dir: 输出文件保存目录
        train_file: 训练集文件名
        val_file: 验证集文件名
        test_file: 测试集文件名
        output_by_type: 是否为每种车牌类型生成单独的数据集文件
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存总体数据集
    train_path = os.path.join(output_dir, train_file)
    with open(train_path, 'w', encoding='utf-8') as f:
        for img_path, label, plate_type in train_data:
            line = f"{img_path} {label}"
            if plate_type:
                line += f" {plate_type}"
            f.write(line + "\n")
    
    val_path = os.path.join(output_dir, val_file)
    with open(val_path, 'w', encoding='utf-8') as f:
        for img_path, label, plate_type in val_data:
            line = f"{img_path} {label}"
            if plate_type:
                line += f" {plate_type}"
            f.write(line + "\n")
    
    test_path = os.path.join(output_dir, test_file)
    with open(test_path, 'w', encoding='utf-8') as f:
        for img_path, label, plate_type in test_data:
            line = f"{img_path} {label}"
            if plate_type:
                line += f" {plate_type}"
            f.write(line + "\n")
    
    print(f"保存总体数据集完成:")
    print(f"训练集: {train_path}")
    print(f"验证集: {val_path}")
    print(f"测试集: {test_path}")
    
    # 如果需要按车牌类型输出单独的数据集文件
    if output_by_type:
        # 按车牌类型分组
        train_by_type = defaultdict(list)
        val_by_type = defaultdict(list)
        test_by_type = defaultdict(list)
        
        # 对训练集进行分组
        for item in train_data:
            _, _, plate_type = item
            train_by_type[plate_type].append(item)
        
        # 对验证集进行分组
        for item in val_data:
            _, _, plate_type = item
            val_by_type[plate_type].append(item)
        
        # 对测试集进行分组
        for item in test_data:
            _, _, plate_type = item
            test_by_type[plate_type].append(item)
        
        # 为每种车牌类型创建单独的数据集文件
        for plate_type in sorted(set(train_by_type.keys()) | set(val_by_type.keys()) | set(test_by_type.keys())):
            # 创建车牌类型目录
            type_dir = os.path.join(output_dir, plate_type.replace(" ", "_"))
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
            
            # 保存该类型的训练集
            train_type_path = os.path.join(type_dir, train_file)
            with open(train_type_path, 'w', encoding='utf-8') as f:
                for img_path, label, pt in train_by_type[plate_type]:
                    f.write(f"{img_path} {label} {pt}\n")
            
            # 保存该类型的验证集
            val_type_path = os.path.join(type_dir, val_file)
            with open(val_type_path, 'w', encoding='utf-8') as f:
                for img_path, label, pt in val_by_type[plate_type]:
                    f.write(f"{img_path} {label} {pt}\n")
            
            # 保存该类型的测试集
            test_type_path = os.path.join(type_dir, test_file)
            with open(test_type_path, 'w', encoding='utf-8') as f:
                for img_path, label, pt in test_by_type[plate_type]:
                    f.write(f"{img_path} {label} {pt}\n")
            
            print(f"\n保存 {plate_type} 类型数据集完成:")
            print(f"训练集: {train_type_path} ({len(train_by_type[plate_type])}样本)")
            print(f"验证集: {val_type_path} ({len(val_by_type[plate_type])}样本)")
            print(f"测试集: {test_type_path} ({len(test_by_type[plate_type])}样本)")


def main():
    # 解析命令行参数
    args = parse_opt()
    
    # 解析划分比例
    ratio_parts = args.ratio.split(':')
    if len(ratio_parts) != 3:
        print(f"错误: 比例参数格式不正确: {args.ratio}，应为'x:y:z'格式")
        return
    
    try:
        ratio = tuple(map(int, ratio_parts))
    except ValueError:
        print(f"错误: 比例参数必须为整数: {args.ratio}")
        return
    
    # 确定数据集根目录
    data_root = args.data_root
    if not data_root:
        data_root = os.path.dirname(os.path.abspath(args.data_txt))
    
    # 确定输出目录
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(args.data_txt))
    
    # 读取数据集索引文件
    data_list = read_data_file(args.data_txt)
    
    # 分析数据集分布情况
    analyze_dataset(data_list)
    
    # 划分数据集
    train_data, val_data, test_data = split_dataset(
        data_list, 
        ratio=ratio, 
        balance=args.balance, 
        by_plate_type=args.by_plate_type,
        random_seed=args.random_seed
    )
    
    # 保存划分后的数据集
    save_split_files(
        train_data, val_data, test_data,
        output_dir,
        args.train_file, args.val_file, args.test_file,
        output_by_type=args.output_by_type
    )


if __name__ == "__main__":
    main()
