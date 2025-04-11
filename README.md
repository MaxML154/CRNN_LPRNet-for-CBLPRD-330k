<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

<div align="center"><a title="" href="https://github.com/zjykzj/crnn-ctc"><img align="center" src="assets/icons/crnn-ctc.svg" alt=""></a></div>

<p align="center">
  «crnn-ctc» implemented CRNN+CTC
<br>

  
ENG/[中文](https://github.com/MaxML154/CRNN_LPRNet-for-CBLPRD-330k/blob/CBLPRD-330k/README_CBLPRD_CN.md "中文介绍")


# CRNN-CTC & LPRNet Chinese License Plate Recognition System (CBLPRD-330k Version)

This fork is a modification from [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc) with improvements and adaptations for the [CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k) dataset.

## Contents
- [Features](#features)
- [Experimental Results](#experimental-results)
- [Installation](#installation)
- [Supported License Plate Types](#supported-license-plate-types)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
  - [Training with CRNN Models](#training-with-crnn-models)
  - [Training with LPRNet Models](#training-with-lprnet-models)
- [Model Evaluation](#model-evaluation)
- [Single Image Prediction](#single-image-prediction)
- [Model Architecture Comparison](#model-architecture-comparison)
  - [CRNN Model](#crnn-model)
  - [LPRNet Model](#lprnet-model)
  - [STNet](#stnet-spatial-transformer-network)
- [Double-Row License Plate Processing](#double-row-license-plate-processing)
- [Data Resampling](#data-resampling)
- [License Plate Skew Correction](#license-plate-skew-correction)
- [Modifications and Improvements](#modifications-and-improvements)
- [References](#references)

## Features

- Recognition support for various license plate types in the CBLPRD-330k dataset
- Two mainstream model architectures:
  - CRNN (Convolutional Recurrent Neural Network)
  - LPRNet (License Plate Recognition Network) and STNet (Spatial Transformer Network)
- Automatic double-row license plate processing (tractor green plates, double-row yellow plates, etc.)
- License plate skew correction to improve recognition accuracy
- Data resampling mechanism to solve the long-tail effect
- License plate type recognition (blue, yellow, green, embassy/consulate plates, etc.)
- Lightweight models suitable for deployment on edge devices

## Experimental Results

The following table shows the performance comparison of different models on the CBLPRD-330k dataset:

| Model | Input Size | Training Time (h) | Model Size (MB)  | Recognition Accuracy (%) |
|-------|------------|--------------|------------|---------------------|
| CRNN + GRU | 128×48 | 2.7 | 65.1 | 99.7 |
| CRNN_Tiny + GRU | 128×48 | 2.5 | 4 | 99.1 |
| CRNN (Standard, with LSTM) | 128×48 | 2.0 | 69.5 | 98.42 |
| CRNN_Tiny (Standard, with LSTM) | 128×48 | 2.0 | 5.1 | 94 |
| CRNN_Tiny + LSTM | 128×48 | 1.4 | 5.1 | 98.5 |
| LPRNet (Standard) | 94×24 | 0.8 | 1.7 | 84.3 |
| LPRNet | 94×24 | 0.5 | 1.8 | 1.8 | 91.2 |
| LPRNetPlus (Standard) | 94×24 | 0.5 | 2.1 | 87.14 |
| LPRNetPlus | 94×24 | 0.52 | 2.13 | 93.6 |
| LPRNet + STNet | 94×24 | - | - | - |
| LPRNetPlus + STNet | 94×24 | - | - | - |

**Note**

- *Training time was measured on a single NVIDIA V100 32G GPU. Model size refers to the size of the .pth file. Recognition accuracy refers to the overall sample-level accuracy.*

- *The number of training set samples is 239478, the number of validation set samples is 68423, and the number of test set samples is 34212.*

- *LPRNet + STNet models are still working on it... Seems need to adjust the setting or structure of STNet so that would achieve proper performance.*

- *Unless otherwise specified, the results shown here are those after adjusting the framework.*

- *The experimental results may vary slightly on different devices and are shown here for reference only.*


## Installation

```bash
# Clone the repository
git clone https://github.com/MaxML154/CRNN_LPRNet-for-CBLPRD-330k.git
cd CRNN_LPRNet-for-CBLPRD-330k

# Install dependencies
pip install -r requirements.txt
```

## Supported License Plate Types

The CBLPRD-330k dataset contains various types of Chinese license plates, which we have adapted for:

1. Standard blue plates (e.g., "粤A·662C1")
2. New energy vehicle plates (e.g., "赣P·DG0218")
3. Single-row yellow plates (e.g., "皖B·14WJM")
4. Double-row yellow plates for trailers (e.g., "藏·SF1G2挂")
5. Tractor green plates (e.g., "湘18EZZG1")
6. Hong Kong and Macau entry plates (e.g., "粤Z·901Z港", "粤Z·TY67澳")
7. Embassy/consulate plates (e.g., "宁479·16领", "434·404使")
8. Driving school plates (with "学" character)
9. Temporary plates (with "临" character)

## Dataset Preparation

The CBLPRD-330k dataset comes with training and validation set listings in the following format:

```
CBLPRD-330k/000208356.jpg 湘18EZZG1 拖拉机绿牌
```

This indicates the image path, license plate number, and license plate type.

If you need to split the dataset by license plate type, you can use the following commands:

```bash
# Split the dataset in a 7:2:1 ratio, balancing by plate type
python split_dataset.py /path/to/data.txt --data-root /path/to/CBLPRD-330k --output-dir /path/to/output --ratio 7:2:1 --by-plate-type --balance

# Create separate dataset files for each plate type
python split_dataset.py /path/to/data.txt --data-root /path/to/CBLPRD-330k --output-dir /path/to/output --ratio 7:2:1 --by-plate-type --balance --output-by-type
```



## Model Training

### Training with CRNN Models

```bash
# Basic training (CRNN_Tiny)
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/ --batch-size 512 --device 0

# Enable all features
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/ --batch-size 512 --device 0 --use-resampling --correct-skew

# Use standard CRNN (non-Tiny version)
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn-cblprd/ --batch-size 256 --device 0 --not-tiny --use-resampling --correct-skew

# Use LSTM instead of GRU
python train_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny_lstm-cblprd/ --batch-size 512 --device 0 --use-lstm --use-resampling --correct-skew
```

### Training with LPRNet Models

```bash
# Use LPRNetPlus (improved version of LPRNet)
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/ --batch-size 512 --device 0

# Use original LPRNet
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet-cblprd-b512/ --batch-size 512 --device 0 --use-origin-block

# Use LPRNetPlus+STNet (Spatial Transformer Network)
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus_stnet-cblprd-b512/ --batch-size 512 --device 0 --add-stnet

# Use LPRNet+STNet
python train_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_stnet-cblprd-b512/ --batch-size 512 --device 0 --use-origin-block --add-stnet
```

### Training Parameters

- `--use-resampling`: Enable data resampling to mitigate the long-tail effect
- `--correct-skew`: Enable license plate skew correction
- `--no-double-process`: Disable double-row license plate processing (enabled by default)

LPRNet-specific parameters:
- `--use-origin-block`: Use original LPRNet implementation instead of LPRNetPlus
- `--add-stnet`: Add STNet spatial transformer network
- `--dropout-rate`: Set dropout rate (default 0.5)

## Model Evaluation

### Evaluating CRNN Models

```bash
# Basic evaluation
python evaluate_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth --batch-size 64 --device 0

# Enable all features
python evaluate_cblprd.py /path/to/CBLPRD-330k/ ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth --batch-size 64 --device 0 --correct-skew
```

### Evaluating LPRNet Models

```bash
# Evaluate LPRNetPlus
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/lprnet_plus-cblprd-b512-best.pth --batch-size 64 --device 0

# Evaluate original LPRNet
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet-cblprd-b512/lprnet-cblprd-b512-best.pth --batch-size 64 --device 0 --use-origin-block

# Evaluate LPRNetPlus+STNet
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus_stnet-cblprd-b512/lprnet_plus_stnet-cblprd-b512-best.pth --batch-size 64 --device 0 --add-stnet

# Save detailed evaluation results
python evaluate_cblprd_lprnet.py /path/to/CBLPRD-330k/ ./runs/lprnet_plus-cblprd-b512/lprnet_plus-cblprd-b512-best.pth --batch-size 64 --device 0 --save-results
```

## Single Image Prediction

```bash
# Basic prediction
python predict_cblprd.py ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth ./sample_image.jpg ./runs/predict/

# Enable all features
python predict_cblprd.py ./runs/crnn_tiny-cblprd/crnn_tiny-cblprd-best.pth ./sample_image.jpg ./runs/predict/ --correct-skew --determine-type
```

### Prediction Parameters

- `--correct-skew`: Enable license plate skew correction
- `--no-double-process`: Disable double-row license plate processing (enabled by default)
- `--determine-type`: Display the recognized plate type

## Model Architecture Comparison

### CRNN Model

CRNN (Convolutional Recurrent Neural Network) combines the advantages of CNN and RNN:
- CNN part extracts image features
- RNN part (GRU or LSTM) models sequence features
- CTC loss function handles sequence recognition without requiring precise alignment

We provide two versions:
- CRNN_Tiny: Lightweight version with fewer parameters, suitable for edge device deployment
- CRNN: Standard version with higher accuracy but greater computational cost

### LPRNet Model

LPRNet is a lightweight network specifically designed for license plate recognition:
- Built with small basic blocks
- No RNN layer, achieving sequence recognition directly through global context
- Smaller network size and faster inference speed

We provide two versions:
- LPRNet: Original implementation
- LPRNetPlus: Improved version with residual connections

### STNet (Spatial Transformer Network)

STNet can be combined with LPRNet:
- Performs spatial transformations on input images before feature extraction
- Automatically learns to correct geometric distortions in input images
- Improves recognition ability for skewed and deformed license plates

## Double-Row License Plate Processing

For double-row license plates (such as tractor green plates, double-row trailer yellow plates), we use a method that joins them into a single-row plate for recognition:

```python
def process_double_layer_plate(img):
    """Process double-row license plates by joining them into a single row"""
    h, w, c = img.shape
    img_upper = img[0:int(5/12*h), :]  # Upper part of the plate
    img_lower = img[int(1/3*h):, :]    # Lower part of the plate
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img
```

Before processing:
![Original double-row plate](https://github.com/MaxML154/CRNN_LPRNet-for-CBLPRD-330k/blob/CBLPRD-330k/assets/plate_origin.jpg)

After processing: 
![After joining](https://github.com/MaxML154/CRNN_LPRNet-for-CBLPRD-330k/blob/CBLPRD-330k/assets/plate_combined.png)

## Data Resampling

To address the uneven distribution of license plate provinces in the dataset, we implemented a data resampling mechanism:

1. Count the number of plates from each province
2. Oversample provinces with sample counts below the median
3. Control the maximum oversampling multiplier to avoid overfitting

## License Plate Skew Correction

Hough transform is used to correct skewed license plates:

1. Detect straight lines in the license plate image
2. Calculate the skew angle of the main horizontal lines
3. Rotate the image based on the angle for correction

## Modifications and Improvements

This project is based on [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc) with the following improvements for the CBLPRD-330k dataset:

1. Added `utils/dataset/cblprd.py` to handle the CBLPRD-330k dataset
2. Added LPRNet and STNet models, providing more model options
3. Implemented automatic double-row license plate processing (enabled by default)
4. Implemented license plate skew correction
5. Added data resampling mechanism
6. Added license plate type recognition functionality
7. Added dataset splitting tool with support for plate type-based splitting

## References

- [zjykzj/crnn-ctc](https://github.com/zjykzj/crnn-ctc)
- [SunlifeV/CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k)
- [we0091234/crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition)
- [LPRNet Paper: LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447)
- [STNet Paper: Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- [guyuealian/blog](https://blog.csdn.net/guyuealian/article/details/128704209) 
