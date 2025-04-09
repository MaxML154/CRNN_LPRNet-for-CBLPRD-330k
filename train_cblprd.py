# -*- coding: utf-8 -*-

"""
@date: 2024/4/8
@file: train_cblprd.py
@author: MaxML154
@description: Training script for CBLPRD-330k dataset using CRNN_Tiny

Usage - Single-GPU training using CRNN_Tiny:
    $ python3 train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd-b512/ --batch-size 512 --device 0
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

from utils.model.crnn import CRNN
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
    parser = argparse.ArgumentParser(description='CBLPRD-330k Training')
    parser.add_argument('data', type=str, help='path to CBLPRD-330k dataset')
    parser.add_argument('output', type=str, help='path to output')
    
    parser.add_argument('--train-txt', type=str, default='train.txt', help='path to train.txt relative to data root')
    parser.add_argument('--val-txt', type=str, default='val.txt', help='path to val.txt relative to data root')
    
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size for all GPUs')
    parser.add_argument('--use-lstm', action='store_true', help='use nn.LSTM instead of nn.GRU')
    parser.add_argument('--not-tiny', action='store_true', help='use this flag to specify non-tiny CRNN')
    
    # 添加新的数据处理选项
    parser.add_argument('--use-resampling', action='store_true', help='use resampling to balance data')
    parser.add_argument('--correct-skew', action='store_true', help='apply skew correction to license plates')
    parser.add_argument('--process-double', action='store_true', help='process double-layer license plates')
    parser.add_argument('--no-resampling', action='store_true', help='disable resampling even if specified')
    parser.add_argument('--no-skew-correction', action='store_true', help='disable skew correction even if specified')
    parser.add_argument('--no-double-process', action='store_true', help='disable double-layer processing even if specified')
    
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
    data_root, batch_size, not_tiny, use_lstm, output = \
        opt.data, opt.batch_size, opt.not_tiny, opt.use_lstm, opt.output
    if RANK in {-1, 0} and not os.path.exists(output):
        os.makedirs(output)
        
    LOGGER.info("=> Create Model")
    # 设定输入形状 (W, H) - 适配CBLPRD-330k数据集的128×48尺寸
    input_shape = (128, 48)
    model = CRNN(in_channel=3, num_classes=len(PLATE_CHARS), cnn_input_height=input_shape[1], 
                is_tiny=not not_tiny, use_gru=not use_lstm).to(device)
    
    if not_tiny:
        model_prefix = 'crnn'
    else:
        model_prefix = "crnn_tiny"
        
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
    process_double = opt.process_double and not opt.no_double_process
    
    # 记录已启用的功能
    enabled_features = []
    if use_resampling:
        enabled_features.append("resampling")
    if correct_skew:
        enabled_features.append("skew correction")
    if process_double:
        enabled_features.append("double-layer processing")
    
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
            
            # 累计损失值
            epoch_loss += loss.item()
            current_train_loss = epoch_loss / (idx + 1)
            
            # 计算训练准确率
            if RANK in {-1, 0}:
                with torch.no_grad():
                    train_acc = train_evaluator.update(outputs.detach().cpu(), targets)
                current_train_acc = train_evaluator.result()
                lr = optimizer.param_groups[0]["lr"]
                
                # 计算批处理速度和预计剩余时间
                elapsed_time = time.time() - epoch_start_time
                iterations_per_sec = (idx + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_iterations = len(train_dataloader) - (idx + 1)
                eta = remaining_iterations / iterations_per_sec if iterations_per_sec > 0 else 0
                
                # 格式化时间
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
                eta_str = str(datetime.timedelta(seconds=int(eta)))
                
                # 更新进度条描述
                info = f"Epoch {epoch} Batch {idx} Training: [{idx+1}/{len(train_dataloader)}] [{elapsed_str}<{eta_str}, {iterations_per_sec:.2f} it/s] LR: {lr:.6f}, Training Loss: {current_train_loss:.6f}, Training ACC: {current_train_acc*100:.3f}%, Validation Loss: {val_loss_value:.6f}, Validation ACC: {val_acc_value*100:.3f}%"
                pbar.set_description(info)
        
        # 计算并记录本epoch的平均训练损失和准确率
        if RANK in {-1, 0}:
            avg_train_loss = epoch_loss / len(train_dataloader)
            avg_train_acc = train_evaluator.result()
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            
            # 每个epoch完成后评估验证集
            model.eval()
            evaluator.reset()
            val_loss = 0.0
            
            # 验证集评估
            LOGGER.info(f"Evaluating on validation set for epoch {epoch}...")
            for idx, (images, targets) in enumerate(val_dataloader):
                images = images.to(device)
                targets = val_dataset.convert(targets)
                target_lengths = torch.IntTensor([len(t) for t in targets]).to(device)
                targets_concat = torch.concat(targets).to(device)
                
                with torch.no_grad():
                    outputs = model(images)
                    # 计算验证损失
                    val_batch_loss = criterion(outputs, targets_concat, target_lengths)
                    val_loss += val_batch_loss.item()
                    
                    # 计算准确率
                    _ = evaluator.update(outputs.cpu(), targets)
            
            # 计算平均验证损失和准确率
            val_loss_value = val_loss / len(val_dataloader)
            val_acc_value = evaluator.result()
            history['val_loss'].append(val_loss_value)
            history['val_acc'].append(val_acc_value)
            
            # 计算完整的epoch时间
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            
            # 输出完整的epoch结果
            LOGGER.info(f"Epoch {epoch} completed in {epoch_time_str} - Training Loss: {avg_train_loss:.6f}, Training ACC: {avg_train_acc*100:.3f}%, Validation Loss: {val_loss_value:.6f}, Validation ACC: {val_acc_value*100:.3f}%")
            
            # 每5个epoch保存模型
            if epoch % 5 == 0 and epoch > 0:
                # 构建包含启用功能的模型名称
                model_name = f"{model_prefix}-cblprd"
                if use_resampling:
                    model_name += "-rs"
                if correct_skew:
                    model_name += "-sk"
                if process_double:
                    model_name += "-db"
                save_path = os.path.join(output, f"{model_name}-b{batch_size}-e{epoch}.pth")
                LOGGER.info(f"Save to {save_path}")
                torch.save(model.state_dict(), save_path)
                
        scheduler.step()
        torch.cuda.empty_cache()
    
    # 训练结束，输出训练总结
    if RANK in {-1, 0}:
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + start_epoch
        LOGGER.info(f"\nTraining Summary:")
        LOGGER.info(f"Best validation accuracy: {best_val_acc*100:.3f}% at epoch {best_epoch}")
    
    LOGGER.info(f'\n{epochs} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')


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
