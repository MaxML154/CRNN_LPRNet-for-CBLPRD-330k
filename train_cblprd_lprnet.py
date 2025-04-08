# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: train_cblprd_lprnet.py
@description: Training script for CBLPRD-330k dataset using LPRNet/LPRNetPlus

Usage - Single-GPU training using LPRNet:
    $ python3 train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet-cblprd-b512/ --batch-size 512 --device 0 --use-origin-block

Usage - Single-GPU training using LPRNetPlus:
    $ python3 train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/ --batch-size 512 --device 0

Usage - Single-GPU training using LPRNet+STNet:
    $ python3 train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_stnet-cblprd-b512/ --batch-size 512 --device 0 --use-origin-block --add-stnet

Usage - Single-GPU training using LPRNetPlus+STNet:
    $ python3 train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus_stnet-cblprd-b512/ --batch-size 512 --device 0 --add-stnet
"""

import argparse
import os.path
import time
import datetime

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

from utils.model.lprnet import LPRNet
from utils.loss import CTCLoss
from utils.evaluator import Evaluator
from utils.torchutil import select_device
from utils.ddputil import smart_DDP
from utils.logger import LOGGER
from utils.general import init_seeds
from utils.dataset.cblprd import CBLPRDDataset, PLATE_CHARS

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt():
    parser = argparse.ArgumentParser(description='CBLPRD-330k Training with LPRNet')
    parser.add_argument('data', type=str, help='path to CBLPRD-330k dataset')
    parser.add_argument('output', type=str, help='path to output')
    
    parser.add_argument('--train-txt', type=str, default='train.txt', help='path to train.txt relative to data root')
    parser.add_argument('--val-txt', type=str, default='val.txt', help='path to val.txt relative to data root')
    
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size for all GPUs')
    
    # LPRNet specific options
    parser.add_argument('--use-origin-block', action='store_true', help='use original LPRNet implementation instead of LPRNetPlus')
    parser.add_argument('--add-stnet', action='store_true', help='add STNet for training and evaluation')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='dropout rate for LPRNet')
    
    # 数据处理选项
    parser.add_argument('--use-resampling', action='store_true', help='use resampling to balance data')
    parser.add_argument('--correct-skew', action='store_true', help='apply skew correction to license plates')
    parser.add_argument('--no-resampling', action='store_true', help='disable resampling even if specified')
    parser.add_argument('--no-skew-correction', action='store_true', help='disable skew correction even if specified')
    parser.add_argument('--no-double-process', action='store_true', help='disable double-layer processing (enabled by default)')
    
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument')
    
    args = parser.parse_args()
    LOGGER.info(f"args: {args}")
    return args


def adjust_learning_rate(lr, warmup_epoch, optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    # Warmup
    lr = lr * float(1 + step + epoch * len_epoch) / (warmup_epoch * len_epoch)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(opt, device):
    data_root, batch_size, use_origin_block, add_stnet, dropout_rate, output = \
        opt.data, opt.batch_size, opt.use_origin_block, opt.add_stnet, opt.dropout_rate, opt.output
    
    if RANK in {-1, 0} and not os.path.exists(output):
        os.makedirs(output)
        
    LOGGER.info("=> Create Model")
    # 设定输入形状 (W, H) - LPRNet使用较小的输入尺寸
    input_shape = (94, 24)
    
    model = LPRNet(in_channel=3, num_classes=len(PLATE_CHARS), 
                  dropout_rate=dropout_rate,
                  use_origin_block=use_origin_block,
                  add_stnet=add_stnet).to(device)
    
    # 确定模型名称前缀
    if use_origin_block:
        model_prefix = 'lprnet'
    else:
        model_prefix = "lprnet_plus"
    if add_stnet:
        model_prefix += '_stnet'
        
    blank_label = 0
    criterion = CTCLoss(blank_label=blank_label).to(device)
    
    learn_rate = 0.001 * WORLD_SIZE
    weight_decay = 1e-5
    LOGGER.info(f"Final learning rate: {learn_rate}, weight decay: {weight_decay}")
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90])
    
    # 确定是否启用各种数据处理选项
    use_resampling = opt.use_resampling and not opt.no_resampling
    correct_skew = opt.correct_skew and not opt.no_skew_correction
    process_double = not opt.no_double_process  # 默认启用双层车牌处理
    
    # 记录已启用的功能
    enabled_features = []
    if use_resampling:
        enabled_features.append("resampling")
    if correct_skew:
        enabled_features.append("skew correction")
    if process_double:
        enabled_features.append("double-layer processing")
    if use_origin_block:
        enabled_features.append("original LPRNet block")
    else:
        enabled_features.append("LPRNetPlus block")
    if add_stnet:
        enabled_features.append("STNet")
    
    if enabled_features:
        LOGGER.info(f"Enabled features: {', '.join(enabled_features)}")
    
    LOGGER.info("=> Load data")
    train_dataset = CBLPRDDataset(data_root, opt.train_txt, is_train=True, input_shape=input_shape,
                                 use_resampling=use_resampling, correct_skew=correct_skew, 
                                 process_double=process_double)
    sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True and sampler is None,
                                 sampler=sampler,
                                 num_workers=4,
                                 drop_last=True,
                                 pin_memory=True)
                                 
    if RANK in {-1, 0}:
        val_dataset = CBLPRDDataset(data_root, opt.val_txt, is_train=False, input_shape=input_shape,
                                  use_resampling=False, correct_skew=correct_skew, 
                                  process_double=process_double)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, 
                                   drop_last=False, pin_memory=True)
                                   
        LOGGER.info("=> Load evaluator")
        evaluator = Evaluator(blank_label=blank_label)
        
    LOGGER.info("=> Start training")
    t0 = time.time()
    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # DDP mode
    cuda = device.type != 'cpu'
    if cuda and RANK != -1:
        model = smart_DDP(model)
        
    epochs = 30
    start_epoch = 1
    warmup_epoch = 5
    
    # 记录训练历史
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': []
    }
    
    # 初始化验证指标
    val_loss_value = 0.0
    val_acc_value = 0.0
    
    for epoch in range(start_epoch, epochs + start_epoch):
        # epoch: start from 1
        model.train()
        if RANK != -1:
            train_dataloader.sampler.set_epoch(epoch)
            
        # 初始化epoch开始时间
        epoch_start_time = time.time()
        
        # 初始化训练指标
        epoch_loss = 0.0
        if RANK in {-1, 0}:
            train_evaluator = Evaluator(blank_label=blank_label)
            
        # 使用tqdm创建进度条
        pbar = train_dataloader
        if LOCAL_RANK in {-1, 0}:
            pbar = tqdm(pbar, total=len(train_dataloader), desc=f"Epoch {epoch}")
        optimizer.zero_grad()
            
        for idx, (images, targets) in enumerate(pbar):
            batch_size = len(images)
            
            targets = train_dataset.convert(targets)
            target_lengths = torch.IntTensor([len(t) for t in targets]).to(device)
            targets_concat = torch.concat(targets).to(device)
            
            with torch.cuda.amp.autocast(amp):
                outputs = model(images.to(device))
                loss = criterion(outputs, targets_concat, target_lengths)
            scaler.scale(loss).backward()
            
            if epoch <= warmup_epoch:
                adjust_learning_rate(learn_rate, warmup_epoch, optimizer, epoch - 1, idx, len(train_dataloader))
                
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            
            # 更新训练指标
            epoch_loss += loss.item()
            if RANK in {-1, 0}:
                # 计算训练批次准确率
                with torch.no_grad():
                    train_acc = train_evaluator.update(outputs.cpu(), targets)
                
                # 更新进度条显示
                lr = optimizer.param_groups[0]["lr"]
                info = f"Epoch:{epoch} Batch:{idx} LR:{lr:.6f} Loss:{loss:.6f} Acc:{train_acc*100:.2f}%"
                
                # 如果有验证集结果，显示上一轮的验证集准确率
                if epoch > start_epoch:
                    info += f" Val:{val_acc_value*100:.2f}%"
                
                pbar.set_description(info)
        
        # 计算平均训练损失和准确率
        if RANK in {-1, 0}:
            train_loss = epoch_loss / len(train_dataloader)
            train_acc = train_evaluator.result()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 计算epoch运行时间
            epoch_time = time.time() - epoch_start_time
            eta = epoch_time * (epochs - epoch + start_epoch)
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            
            # 打印训练结果
            LOGGER.info(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s (ETA: {eta_str})")
            LOGGER.info(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc*100:.2f}%")
            
            # 每个epoch结束后，在验证集上评估模型
            model.eval()
            evaluator.reset()
            val_loss = 0.0
            
            # 使用tqdm创建验证进度条
            val_pbar = tqdm(val_dataloader, desc=f"Validating")
            
            for idx, (images, targets) in enumerate(val_pbar):
                targets = val_dataset.convert(targets)
                target_lengths = torch.IntTensor([len(t) for t in targets]).to(device)
                targets_concat = torch.concat(targets).to(device)
                
                with torch.no_grad():
                    outputs = model(images.to(device))
                    loss = criterion(outputs, targets_concat, target_lengths)
                    
                val_loss += loss.item()
                val_acc = evaluator.update(outputs.cpu(), targets)
                val_pbar.set_description(f"Val Batch:{idx} Loss:{loss:.6f} Acc:{val_acc*100:.2f}%")
            
            # 计算平均验证损失和准确率
            val_loss_value = val_loss / len(val_dataloader)
            val_acc_value = evaluator.result()
            history['val_loss'].append(val_loss_value)
            history['val_acc'].append(val_acc_value)
            
            # 打印验证结果
            LOGGER.info(f"Val Loss: {val_loss_value:.6f}, Val Acc: {val_acc_value*100:.2f}%")
            
            # 保存检查点
            if epoch % 5 == 0 or epoch == epochs:
                save_path = os.path.join(output, f"{model_prefix}-cblprd-b{batch_size}-e{epoch}.pth")
                LOGGER.info(f"Saving checkpoint to {save_path}")
                torch.save(model.state_dict(), save_path)
                
                # 保存最佳模型（根据验证准确率）
                if val_acc_value == max(history['val_acc']):
                    best_path = os.path.join(output, f"{model_prefix}-cblprd-b{batch_size}-best.pth")
                    LOGGER.info(f"Saving best model to {best_path} (Val Acc: {val_acc_value*100:.2f}%)")
                    torch.save(model.state_dict(), best_path)
        
        # 更新学习率
        scheduler.step()
        torch.cuda.empty_cache()
    
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epochs} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        LOGGER.info(f"Best validation accuracy: {max(history['val_acc'])*100:.2f}%")


def main(opt):
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with Multi-GPU DDP training'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    init_seeds(opt.seed + 1 + RANK, deterministic=False)
    train(opt, device)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 