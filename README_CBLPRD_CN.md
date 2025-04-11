# CRNN-CTC & LPRNet 中国车牌识别系统 (CBLPRD-330k适配版)

[Eng](https://github.com/MaxML154/CRNN_LPRNet-for-CBLPRD-330k/blob/CBLPRD-330k/README.md "README.md")/中文

## 目录
- [特点](#特点)
- [实验结果](#实验结果)
- [安装](#安装)
- [支持的车牌类型](#支持的车牌类型)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
  - [使用CRNN模型训练](#使用crnn模型训练)
  - [使用LPRNet模型训练](#使用lprnet模型训练)
- [模型评估](#模型评估)
- [单张图片预测](#单张图片预测)
- [模型架构对比](#模型架构对比)
  - [CRNN模型](#crnn模型)
  - [LPRNet模型](#lprnet模型)
  - [STNet](#stnet-空间变换网络)
- [双层车牌处理](#双层车牌处理)
- [数据重采样](#数据重采样)
- [车牌倾斜矫正](#车牌倾斜矫正)
- [修改和改进](#修改和改进)
- [参考](#参考)

本项目是基于[zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)改进，针对[CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k)数据集进行适配的车牌识别系统。

## 特点

- 支持CBLPRD-330k数据集中的各种车牌类型识别
- 提供两种主流模型架构:
  - CRNN (卷积循环神经网络)
  - LPRNet (轻量级车牌识别网络)和STNet (空间变换网络)
- 双层车牌自动处理（拖拉机绿牌、双层黄牌挂车等）
- 实现车牌倾斜矫正，提高识别准确率
- 数据重采样机制，解决长尾效应问题
- 支持车牌类型识别（蓝牌、黄牌、绿牌、使领馆车牌等）
- 轻量级模型，适合部署在边缘设备

## 实验结果

下表显示了不同模型在CBLPRD-330k数据集上的性能比较：

| 模型 | 输入尺寸 | 训练时间 （小时h） | 模型大小 （MB） |  识别准确率 （%） |
|-------|------------|--------------|------------|---------------------|
| CRNN + GRU | 128×48 | 2.7 | 65.1 | 99.7 |
| CRNN_Tiny + GRU | 128×48 | 2.5 | 4 | 99.1 |
| CRNN (默认, 搭配 LSTM) | 128×48 | 2.0 | 69.5 | 98.42 |
| CRNN_Tiny (默认, 搭配 LSTM) | 128×48 | 2.0 | 5.1 | 94 |
| CRNN_Tiny + LSTM | 128×48 | 1.4 | 5.1 | 98.5 |
| LPRNet (默认) | 94×24 | 0.8 | 1.7 | 84.3 |
| LPRNet | 94×24 | 0.5 | 1.8 | 91.2 |
| LPRNetPlus (默认) | 94×24 | 0.5 | 2.1 | 87.14 |
| LPRNetPlus | 94×24 | 0.52 | 2.13 | 93.6 |
| LPRNet + STNet | 94×24 | - | - | - |
| LPRNetPlus + STNet | 94×24 | - | - | - |

*注：*

- *训练时间在单个NVIDIA V100 32G GPU上测量；模型大小指.pth文件的大小；识别准确率指整体样本级准确率。*

- *训练集数量为239478，验证集数量为68423，测试集数量为34212。*

- *所有实验都在批次大小为512，迭代次数为30进行，训练时存在5次迭代的预热，学习率设置为0.001，权重衰减设置为1e-05使用Adam优化器，同时使用MultiStepLR进行学习率调度，默认开启混合精度训练。*

- *LPRNet + STNet 结合的模型仍在调整中，需要修改原版STNet的部分内容才可能会达到一个较好的性能。*

- *如无特别说明，此处所展示的结果为调整框架后的结果。*

- *实验结果会在不同设备上存在些许出入，此处展示仅供参考。*

## 安装

```bash
# 克隆仓库
git clone https://github.com/MaxML154/CRNN_LPRNet-for-CBLPRD-330k.git
cd CRNN_LPRNet-for-CBLPRD-330k

# 安装依赖
pip install -r requirements.txt
```

## 支持的车牌类型

CBLPRD-330k数据集包含多种类型的中国车牌，我们已适配以下类型：

1. 普通蓝牌（如"粤A·662C1"）
2. 新能源车牌（如"赣P·DG0218"）
3. 单层黄牌（如"皖B·14WJM"）
4. 双层黄牌挂车（如"藏·SF1G2挂"）
5. 拖拉机绿牌（如"湘18EZZG1"）
6. 港澳入出内地车牌（如"粤Z·901Z港"、"粤Z·TY67澳"）
7. 使领馆车牌（如"宁479·16领"、"434·404使"）
8. 教练车牌（含"学"字样）
9. 临时车牌（含"临"字样）

## 数据集准备

CBLPRD-330k数据集已由原作者提供了训练集和验证集列表，格式为：

```
CBLPRD-330k/000208356.jpg 湘18EZZG1 拖拉机绿牌
```

表示图片路径、车牌号和车牌类型。

如果需要按车牌类型划分数据集，可以使用以下命令：

```bash
# 按7:2:1比例划分数据集，按车牌类型平衡数据
python split_dataset.py /path/to/data.txt --data-root /path/to/CBLPRD-330k --output-dir /path/to/output --ratio 7:2:1 --by-plate-type --balance

# 为每种车牌类型创建单独的数据集文件
python split_dataset.py /path/to/data.txt --data-root /path/to/CBLPRD-330k --output-dir /path/to/output --ratio 7:2:1 --by-plate-type --balance --output-by-type
```



## 模型训练

### 使用CRNN模型训练

```bash
# 基本训练 (CRNN_Tiny)
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/ --batch-size 512 --device 0

# 启用所有特性
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/ --batch-size 512 --device 0 --use-resampling --correct-skew

# 使用标准CRNN (非Tiny版)
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn-cblprd/ --batch-size 256 --device 0 --not-tiny --use-resampling --correct-skew

# 使用LSTM代替GRU
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny_lstm-cblprd/ --batch-size 512 --device 0 --use-lstm --use-resampling --correct-skew
```

### 使用LPRNet模型训练

```bash
# 使用LPRNetPlus (改进版LPRNet)
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/ --batch-size 512 --device 0

# 使用原始LPRNet
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet-cblprd-b512/ --batch-size 512 --device 0 --use-origin-block

# 使用LPRNetPlus+STNet (空间变换网络)
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus_stnet-cblprd-b512/ --batch-size 512 --device 0 --add-stnet

# 使用LPRNet+STNet
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_stnet-cblprd-b512/ --batch-size 512 --device 0 --use-origin-block --add-stnet
```

### 训练参数说明

- `--use-resampling`: 启用数据重采样，缓解长尾效应
- `--correct-skew`: 启用车牌倾斜矫正
- `--no-double-process`: 关闭双层车牌处理 (默认启用)

LPRNet特有参数:
- `--use-origin-block`: 使用原始LPRNet实现替代LPRNetPlus
- `--add-stnet`: 添加STNet空间变换网络
- `--dropout-rate`: 设置Dropout比率 (默认0.5)

## 模型评估

### 评估CRNN模型

```bash
# 基本评估
python evaluate_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth --batch-size 64 --device 0

# 启用所有特性
python evaluate_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth --batch-size 64 --device 0 --correct-skew
```

### 评估LPRNet模型

```bash
# 评估LPRNetPlus
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/lprnet_plus-cblprd-b512-best.pth --batch-size 64 --device 0

# 评估原始LPRNet
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet-cblprd-b512/lprnet-cblprd-b512-best.pth --batch-size 64 --device 0 --use-origin-block

# 评估LPRNetPlus+STNet
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus_stnet-cblprd-b512/lprnet_plus_stnet-cblprd-b512-best.pth --batch-size 64 --device 0 --add-stnet

# 保存详细评估结果
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/lprnet_plus-cblprd-b512-best.pth --batch-size 64 --device 0 --save-results
```

## 单张图片预测

```bash
# 基本预测
python predict_cblprd.py ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth ./sample_image.jpg ./runs/predict/

# 启用所有特性
python predict_cblprd.py ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth ./sample_image.jpg ./runs/predict/ --correct-skew --determine-type
```

### 预测参数说明

- `--correct-skew`: 启用车牌倾斜矫正
- `--no-double-process`: 关闭双层车牌处理 (默认启用)
- `--determine-type`: 显示识别出的车牌类型

## 模型架构对比

### CRNN模型

CRNN (Convolutional Recurrent Neural Network) 结合了CNN和RNN的优势:
- CNN部分提取图像特征
- RNN部分(GRU或LSTM)对序列特征建模
- CTC损失函数处理无需精确对齐的序列识别问题

我们提供了两个版本:
- CRNN_Tiny: 轻量级版本，参数更少，适合边缘设备部署
- CRNN: 标准版本，精度更高，但计算开销更大

### LPRNet模型

LPRNet是专为车牌识别设计的轻量级网络:
- 采用小型基础块(Small Basic Block)构建
- 无需RNN层，直接通过全局上下文实现序列识别
- 更小的网络规模和更快的推理速度

我们提供了两个版本:
- LPRNet: 原始实现
- LPRNetPlus: 增加了残差连接的改进版本

### STNet (空间变换网络)

STNet可以与LPRNet结合使用:
- 在特征提取前对输入图像进行空间变换
- 自动学习矫正输入图像的几何失真
- 提高对倾斜、变形车牌的识别能力

## 双层车牌处理

对于双层车牌（如拖拉机绿牌、双层黄牌挂车），我们采用拼接成单层车牌的方式进行识别：

```python
def process_double_layer_plate(img):
    """处理双层车牌，将双层车牌拼接成单层车牌"""
    h, w, c = img.shape
    img_upper = img[0:int(5/12*h), :]  # 上半部分车牌
    img_lower = img[int(1/3*h):, :]    # 下半部分车牌
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img
```

拼接处理前:
![双层车牌原图](/assets/plate_origin.jpg)

拼接处理后: 
![拼接处理后](assets/plate_combined.png)

## 数据重采样

为解决数据集中车牌省份分布不均的问题，我们实现了数据重采样机制：

1. 统计各省份车牌数量
2. 对样本数少于中位数的省份进行过采样
3. 控制最大过采样倍数，避免过拟合

## 车牌倾斜矫正

使用霍夫变换对倾斜车牌进行矫正：

1. 检测车牌图像中的直线
2. 计算主要水平线的倾斜角度
3. 根据角度旋转图像进行矫正

## 修改和改进

本项目基于[zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)，针对CBLPRD-330k数据集进行了以下改进：

1. 增加`utils/dataset/cblprd.py`处理CBLPRD-330k数据集
2. 添加LPRNet和STNet模型，提供更多模型选择
3. 实现双层车牌自动处理功能（默认启用）
4. 实现车牌倾斜矫正
5. 添加数据重采样机制
6. 增加车牌类型判断功能
7. 添加数据集划分工具，支持按车牌类型划分

## 参考

- [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)
- [SunlifeV/CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k)
- [we0091234/crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition)
- [LPRNet论文：LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447)
- [STNet论文：Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- [guyuealian/blog](https://blog.csdn.net/guyuealian/article/details/128704209) 
