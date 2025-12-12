#!/usr/bin/env python3
"""
Single Image Inference with DnCNN

Denoise a single image or all images in a directory.

Usage:
    # Denoise single image
    python inference.py --input noisy.jpg --output denoised.jpg
    
    # Denoise all images in directory
    python inference.py --input_dir noisy_images/ --output_dir denoised_images/
    
    # Use fine-tuned model
    python inference.py --input noisy.jpg --model_path checkpoints/best.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model import DnCNN, load_pretrained_dncnn
from utils import download_pretrained_model, tensor_to_image, image_to_tensor, calculate_psnr


def denoise_image(
    model: torch.nn.Module,
    image_path: str,
    device: str = 'cpu',
    grayscale: bool = False,
    resize: int = 256
) -> np.ndarray:
    """
    Denoise a single image
    
    Args:
        model: DnCNN model
        image_path: Path to input image
        device: Device to use
        grayscale: Process as grayscale
        resize: Resize to this size (0 = no resize)
        
    Returns:
        Denoised image as numpy array
    """
    # Load image
    img = Image.open(image_path)
    original_mode = img.mode
    
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    # Resize if specified
    if resize > 0:
        img = img.resize((resize, resize), Image.LANCZOS)
    
    # Convert to tensor
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = image_to_tensor(img_np, device)
    
    # Denoise
    model.eval()
    with torch.no_grad():
        denoised = model(img_tensor)
        denoised = torch.clamp(denoised, 0, 1)
    
    # Convert back to numpy
    denoised_np = tensor_to_image(denoised)
    
    return denoised_np


def main():
    parser = argparse.ArgumentParser(
        description='Denoise images with DnCNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Denoise single image with pre-trained model
  python inference.py --input noisy.jpg --output clean.jpg
  
  # Denoise with specific noise level model
  python inference.py --input noisy.jpg --noise_level 50
  
  # Denoise directory with fine-tuned model
  python inference.py --input_dir noisy/ --output_dir clean/ --model_path checkpoints/best.pth
        """
    )
    
    # Input/output arguments
    parser.add_argument('--input', type=str, default=None,
                        help='Input image path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory (for batch processing)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (for batch processing)')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--noise_level', type=int, default=25,
                        help='Noise level for pre-trained model')
    parser.add_argument('--blind', action='store_true',
                        help='Use blind denoising model')
    parser.add_argument('--grayscale', action='store_true',
                        help='Process as grayscale')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize images to this size (default: 256). Use 0 for no resize.')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, or auto)')
    parser.add_argument('--compare', action='store_true',
                        help='Save side-by-side comparison')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input_dir must be specified")
    
    # Determine device (supports CUDA, MPS for Mac, and CPU)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Apple Metal GPU
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    in_channels = 1 if args.grayscale else 3
    
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = load_pretrained_dncnn(args.model_path, device, in_channels)
    else:
        # Select appropriate model
        if args.grayscale:
            if args.blind:
                model_name = 'dncnn_gray_blind'
            else:
                model_name = f'dncnn_{args.noise_level}'
        else:
            # COLOR images - must use color model
            model_name = 'dncnn_color_blind'
            if not args.blind:
                print(f"  Note: Using 'dncnn_color_blind' for color images")
        
        print(f"Using pre-trained model: {model_name}")
        model_path = download_pretrained_model(model_name, save_dir='pretrained')
        model = load_pretrained_dncnn(model_path, device, in_channels)
    
    # Process single image
    if args.input:
        input_path = Path(args.input)
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_denoised{input_path.suffix}"
        
        print(f"Processing {input_path}...")
        denoised = denoise_image(model, str(input_path), device, args.grayscale, args.resize)
        
        # Save result
        if args.compare:
            # Create side-by-side comparison
            original = np.array(Image.open(input_path).convert('RGB' if not args.grayscale else 'L'))
            if args.grayscale:
                denoised_rgb = np.stack([denoised] * 3, axis=-1) if denoised.ndim == 2 else denoised
                original_rgb = np.stack([original] * 3, axis=-1) if original.ndim == 2 else original
            else:
                denoised_rgb = denoised
                original_rgb = original
            
            comparison = np.hstack([original_rgb, denoised_rgb])
            Image.fromarray(comparison).save(output_path)
        else:
            Image.fromarray(denoised).save(output_path)
        
        print(f"Saved to {output_path}")
    
    # Process directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_denoised"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(images)} images in {input_dir}")
        
        for img_path in tqdm(images, desc="Processing"):
            denoised = denoise_image(model, str(img_path), device, args.grayscale, args.resize)
            
            output_path = output_dir / img_path.name
            Image.fromarray(denoised).save(output_path)
        
        print(f"Saved {len(images)} images to {output_dir}")


if __name__ == '__main__':
    main()

