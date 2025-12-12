# Intelligent Image Enhancement

A PyTorch-based image denoising and enhancement framework using Deep Convolutional Neural Networks (DnCNN). This project provides a complete pipeline for image denoising, including pre-trained models, custom training, evaluation, and inference capabilities.

## Features

- **Pre-trained DnCNN Models**: Access to KAIR pre-trained models for various noise levels (15, 25, 50) and blind denoising
- **Custom Training**: Fine-tune models on your own datasets with paired distorted/clean images
- **Flexible Inference**: Denoise single images or batch process entire directories
- **Comprehensive Evaluation**: Calculate PSNR, SSIM, and other quality metrics
- **Low-Light Enhancement**: Additional module for low-light image enhancement
- **Easy Setup**: Automated model downloading and environment verification

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Setup and Testing](#setup-and-testing)
  - [Inference (Image Denoising)](#inference-image-denoising)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Dataset Format](#dataset-format)
- [Acknowledgments](#acknowledgments)

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/shrenik-jain/intelligent-image-enhancement.git
cd intelligent-image-enhancement
```

2. **Install dependencies**:
```bash
pip install -r requirements_training.txt
```

3. **Verify installation**:
```bash
python setup_and_test.py
```

This script will:
- Check all dependencies
- Download a pre-trained model
- Run a quick test to verify everything works
- Create sample data structure

## ⚡ Quick Start

### Denoise a Single Image

```bash
python inference.py --input noisy_image.jpg --output denoised_image.jpg
```

### Denoise Multiple Images

```bash
python inference.py --input_dir noisy_images/ --output_dir denoised_images/
```

### Use Specific Noise Level Model

```bash
# For images with noise level ~25
python inference.py --input noisy.jpg --noise_level 25 --output clean.jpg
```

## Project Structure

```
intelligent-image-enhancement/
├── model.py                      # DnCNN model architecture
├── train.py                      # Training script
├── inference.py                  # Image denoising inference
├── evaluate_pretrained.py        # Model evaluation
├── dataset.py                    # Dataset loaders
├── utils.py                      # Helper functions and metrics
├── setup_and_test.py            # Setup verification script
├── requirements_training.txt     # Python dependencies
├── checkpoints/                  # Saved model checkpoints
│   ├── patch256/                # Models trained on 256×256 patches
│   ├── patch1024/               # Models trained on 1024×1024 patches
│   └── patch64/                 # Models trained on 64×64 patches
├── pretrained/                   # Pre-trained KAIR models
│   ├── dncnn_25.pth             # Noise level 25
│   └── dncnn_color_blind.pth    # Blind denoising
├── image_processing/             # Additional enhancement modules
│   └── low_light_enhancement.py # Low-light image enhancement
└── results/                      # Evaluation results and outputs
    └── denoised/                # Denoised images
```

## Usage

### Setup and Testing

Before starting, verify your environment:

```bash
python setup_and_test.py
```

This will check dependencies, download pre-trained models, and run basic tests.

### Inference (Image Denoising)

#### Single Image

```bash
python inference.py \
    --input noisy_image.jpg \
    --output denoised_image.jpg \
    --noise_level 25
```

#### Batch Processing

```bash
python inference.py \
    --input_dir path/to/noisy/images/ \
    --output_dir path/to/output/ \
    --noise_level 25
```

#### Using Fine-tuned Model

```bash
python inference.py \
    --input noisy.jpg \
    --output clean.jpg \
    --model_path checkpoints/patch256/dncnn_finetuned_best.pth
```

#### Options

- `--input`: Input image file
- `--input_dir`: Input directory with multiple images
- `--output`: Output file path
- `--output_dir`: Output directory
- `--model_path`: Path to custom trained model
- `--noise_level`: Noise level (15, 25, 50) for pre-trained models
- `--grayscale`: Process as grayscale image
- `--resize`: Resize images to specified size

### Training

Train or fine-tune DnCNN on your custom dataset:

#### Basic Training

```bash
python train.py \
    --data_dir path/to/training/data \
    --epochs 50 \
    --batch_size 16
```

#### Advanced Training

```bash
python train.py \
    --data_dir path/to/training/data \
    --val_dir path/to/validation/data \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --patch_size 256 \
    --save_dir checkpoints/my_model
```

#### Resume Training

```bash
python train.py \
    --data_dir path/to/training/data \
    --resume checkpoints/my_model/dncnn_finetuned_epoch_20.pth
```

#### Training Options

- `--data_dir`: Training data directory (required)
- `--val_dir`: Validation data directory (optional)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.0001)
- `--patch_size`: Patch size for training (64, 256, 1024)
- `--pretrained_model`: Pre-trained model to start from
- `--save_dir`: Directory to save checkpoints
- `--resume`: Resume from checkpoint

### Evaluation

Evaluate model performance on test dataset:

#### Evaluate Pre-trained Model

```bash
python evaluate_pretrained.py \
    --data_dir path/to/test/data \
    --noise_level 25
```

#### Evaluate Fine-tuned Model

```bash
python evaluate_pretrained.py \
    --data_dir path/to/test/data \
    --model_path checkpoints/patch256/dncnn_finetuned_best.pth \
    --save_results
```

#### Evaluation Options

- `--data_dir`: Test data directory
- `--model_path`: Path to model checkpoint
- `--noise_level`: Noise level for pre-trained models
- `--save_results`: Save denoised images
- `--results_dir`: Directory for results (default: results/)


## Dataset Format

### Training Data Structure

Your dataset should be organized as follows:

```
data/
├── train/
│   ├── distorted/          # Noisy/distorted images
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ground_truth/       # Clean ground truth images
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── distorted/
    │   └── ...
    └── ground_truth/
        └── ...
```

## Acknowledgments

- **DnCNN Paper**: Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE Transactions on Image Processing.
- **KAIR**: Pre-trained models from [KAIR (Kai Zhang's Image Restoration)](https://github.com/cszn/KAIR)