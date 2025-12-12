"""
Utility Functions for DnCNN Training and Evaluation

Includes:
- Image quality metrics (PSNR, SSIM)
- Model download utilities
- Visualization helpers
"""

import os
import math
from pathlib import Path
from typing import Tuple, Optional, Union
import urllib.request
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# IMAGE QUALITY METRICS
# ============================================================================

def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    max_val: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image (prediction)
        img2: Second image (ground truth)
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)
        
    Returns:
        PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * math.log10((max_val ** 2) / mse)
    return psnr


def _ssim_single(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0, window_size: int = 11) -> float:
    """Calculate SSIM for a single image pair (HWC or HW format)"""
    # Convert to grayscale for SSIM calculation if color
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1_gray = 0.299 * img1[:,:,0] + 0.587 * img1[:,:,1] + 0.114 * img1[:,:,2]
        img2_gray = 0.299 * img2[:,:,0] + 0.587 * img2[:,:,1] + 0.114 * img2[:,:,2]
    elif img1.ndim == 3 and img1.shape[2] == 1:
        img1_gray = img1.squeeze()
        img2_gray = img2.squeeze()
    else:
        img1_gray = img1
        img2_gray = img2
    
    # SSIM constants
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Gaussian window
    def gaussian_window(size, sigma=1.5):
        x = np.arange(size) - size // 2
        gauss = np.exp(-x**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        return np.outer(gauss, gauss)
    
    window = gaussian_window(window_size)
    
    # Compute means using convolution
    from scipy.ndimage import convolve
    
    mu1 = convolve(img1_gray, window, mode='reflect')
    mu2 = convolve(img2_gray, window, mode='reflect')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = convolve(img1_gray ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = convolve(img2_gray ** 2, window, mode='reflect') - mu2_sq
    sigma12 = convolve(img1_gray * img2_gray, window, mode='reflect') - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    max_val: float = 1.0,
    window_size: int = 11
) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Handles both single images and batches.
    
    Args:
        img1: First image (prediction) - can be (H,W), (C,H,W), or (B,C,H,W)
        img2: Second image (ground truth) - same format as img1
        max_val: Maximum pixel value
        window_size: Size of the sliding window
        
    Returns:
        SSIM value (0 to 1, higher is better) - averaged over batch if batched
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batched input (B, C, H, W)
    if img1.ndim == 4:
        batch_size = img1.shape[0]
        ssim_values = []
        for i in range(batch_size):
            single_img1 = img1[i].transpose(1, 2, 0)  # CHW -> HWC
            single_img2 = img2[i].transpose(1, 2, 0)
            ssim_values.append(_ssim_single(single_img1, single_img2, max_val, window_size))
        return float(np.mean(ssim_values))
    
    # Single image (C, H, W) -> convert to (H, W, C)
    if img1.ndim == 3 and img1.shape[0] in [1, 3]:
        img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3 and img2.shape[0] in [1, 3]:
        img2 = img2.transpose(1, 2, 0)
    
    return _ssim_single(img1, img2, max_val, window_size)


def calculate_metrics(
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    max_val: float = 1.0
) -> dict:
    """
    Calculate all image quality metrics
    
    Args:
        prediction: Predicted/restored image
        ground_truth: Ground truth image
        max_val: Maximum pixel value
        
    Returns:
        Dictionary with PSNR and SSIM values
    """
    psnr = calculate_psnr(prediction, ground_truth, max_val)
    ssim = calculate_ssim(prediction, ground_truth, max_val)
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }


# ============================================================================
# MODEL DOWNLOAD UTILITIES
# ============================================================================

# Pre-trained model URLs from KAIR repository
# IMPORTANT: dncnn_15/25/50 are GRAYSCALE only! Use dncnn_color_blind for color images.
PRETRAINED_MODELS = {
    'dncnn_15': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_15.pth',
        'description': 'DnCNN for noise σ=15 (GRAYSCALE ONLY)',
        'channels': 1,
        'layers': 17
    },
    'dncnn_25': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth',
        'description': 'DnCNN for noise σ=25 (GRAYSCALE ONLY)',
        'channels': 1,
        'layers': 17
    },
    'dncnn_50': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_50.pth',
        'description': 'DnCNN for noise σ=50 (GRAYSCALE ONLY)',
        'channels': 1,
        'layers': 17
    },
    'dncnn_gray_blind': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth',
        'description': 'DnCNN blind denoising σ=0-55 (GRAYSCALE)',
        'channels': 1,
        'layers': 20
    },
    'dncnn_color_blind': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth',
        'description': 'DnCNN blind denoising σ=0-55 (COLOR/RGB)',
        'channels': 3,
        'layers': 20
    },
    'dncnn3': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn3.pth',
        'description': 'DnCNN for JPEG deblocking (GRAYSCALE)',
        'channels': 1,
        'layers': 17
    }
}


def get_model_for_channels(channels=3):
    """
    Get the appropriate model name for the given number of channels
    
    Args:
        channels: 1 for grayscale, 3 for color
        
    Returns:
        Recommended model name
    """
    if channels == 3:
        return 'dncnn_color_blind'  # Only color model available
    else:
        return 'dncnn_gray_blind'  # Best grayscale model (handles various noise levels)


def download_pretrained_model(
    model_name: str = 'dncnn_25',
    save_dir: str = 'pretrained',
    force_download: bool = False
) -> str:
    """
    Download a pre-trained DnCNN model
    
    Args:
        model_name: Name of the model (see PRETRAINED_MODELS)
        save_dir: Directory to save the model
        force_download: Re-download even if file exists
        
    Returns:
        Path to the downloaded model file
    """
    if model_name not in PRETRAINED_MODELS:
        available = ', '.join(PRETRAINED_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_info = PRETRAINED_MODELS[model_name]
    url = model_info['url']
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine save path
    filename = f"{model_name}.pth"
    save_path = save_dir / filename
    
    # Download if needed
    if not save_path.exists() or force_download:
        print(f"Downloading {model_name} from {url}...")
        print(f"Description: {model_info['description']}")
        
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Downloaded to {save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    else:
        print(f"Model already exists at {save_path}")
    
    return str(save_path)


def list_available_models():
    """Print list of available pre-trained models"""
    print("\nAvailable pre-trained DnCNN models:")
    print("-" * 70)
    print(f"  {'Model':<20} {'Channels':<10} {'Description'}")
    print("-" * 70)
    for name, info in PRETRAINED_MODELS.items():
        ch = 'Color' if info['channels'] == 3 else 'Gray'
        print(f"  {name:<20} {ch:<10} {info['description']}")
    print("-" * 70)
    print("\n  NOTE: For COLOR images, use 'dncnn_color_blind'")
    print("        For GRAYSCALE images, use 'dncnn_gray_blind' or 'dncnn_25'")
    print("-" * 70)


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy image
    
    Args:
        tensor: Image tensor (B, C, H, W) or (C, H, W)
        
    Returns:
        Numpy image (H, W, C) in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    img = tensor.detach().cpu().numpy()
    
    if img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)
    
    if img.shape[2] == 1:
        img = img.squeeze(2)
    
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def image_to_tensor(img: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Convert numpy image to PyTorch tensor
    
    Args:
        img: Numpy image (H, W) or (H, W, C) in range [0, 255] or [0, 1]
        device: Target device
        
    Returns:
        Image tensor (1, C, H, W) in range [0, 1]
    """
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    
    if img.ndim == 2:
        img = img[np.newaxis, np.newaxis, ...]
    else:
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
    
    return torch.from_numpy(img).to(device)


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image file"""
    img = tensor_to_image(tensor)
    Image.fromarray(img).save(path)
    print(f"Saved image to {path}")


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Load image file as numpy array"""
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.5, decay_epoch=30):
    """Decay learning rate by decay_rate every decay_epoch epochs"""
    lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test metrics
    img1 = np.random.rand(100, 100, 3).astype(np.float32)
    img2 = img1 + np.random.randn(100, 100, 3).astype(np.float32) * 0.1
    img2 = np.clip(img2, 0, 1)
    
    metrics = calculate_metrics(img1, img2)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    
    # List available models
    list_available_models()
    
    print("\nUtility tests passed!")

