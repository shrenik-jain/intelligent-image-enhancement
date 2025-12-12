#!/usr/bin/env python3
"""
Evaluate Pre-trained or Fine-tuned DnCNN Model

This script evaluates the model on your test dataset and computes metrics.

Usage:
    # Evaluate pre-trained model (downloads automatically)
    python evaluate_pretrained.py --data_dir data/test --noise_level 25
    
    # Evaluate fine-tuned model
    python evaluate_pretrained.py --data_dir data/test --model_path checkpoints/dncnn_finetuned_best.pth
    
    # Evaluate and save denoised images
    python evaluate_pretrained.py --data_dir data/test --save_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import load_pretrained_dncnn, get_model_info
from dataset import PairedImageDataset, SingleImageDataset
from utils import (
    calculate_metrics, download_pretrained_model, list_available_models,
    tensor_to_image, save_image, PRETRAINED_MODELS
)


def evaluate_on_dataset(
    model: torch.nn.Module,
    data_dir: str,
    device: str = 'cpu',
    save_results: bool = False,
    results_dir: str = 'results',
    grayscale: bool = False,
    resize: int = 256
) -> dict:
    """
    Evaluate model on a paired dataset
    
    Args:
        model: DnCNN model
        data_dir: Directory containing 'distorted' and 'ground_truth' subdirs
        device: 'cpu' or 'cuda'
        save_results: Whether to save denoised images
        results_dir: Directory to save results
        grayscale: Use grayscale mode
        
    Returns:
        Dictionary with evaluation metrics
    """
    distorted_dir = os.path.join(data_dir, 'distorted')
    gt_dir = os.path.join(data_dir, 'ground_truth')
    
    if not os.path.exists(distorted_dir) or not os.path.exists(gt_dir):
        raise ValueError(
            f"Data directory must contain 'distorted' and 'ground_truth' subdirectories.\n"
            f"Expected:\n  {distorted_dir}\n  {gt_dir}"
        )
    
    # Create dataset with resize
    resize_size = resize if resize > 0 else None
    dataset = PairedImageDataset(
        distorted_dir, gt_dir,
        resize=resize_size,
        patch_size=None,
        augment=False,
        grayscale=grayscale
    )
    
    # Create results directory if saving
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        denoised_dir = results_path / 'denoised'
        denoised_dir.mkdir(exist_ok=True)
    
    # Evaluate
    model.eval()
    all_metrics = []
    
    print(f"\nEvaluating on {len(dataset)} images...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing"):
            distorted, ground_truth, filename = dataset[i]
            
            # Add batch dimension and move to device
            distorted = distorted.unsqueeze(0).to(device)
            ground_truth = ground_truth.unsqueeze(0).to(device)
                
            
            # Forward pass
            denoised = model(distorted)
            
            # Clamp output to valid range
            denoised = torch.clamp(denoised, 0, 1)
            
            # Calculate metrics
            # Metrics BEFORE denoising (distorted vs ground truth)
            metrics_before = calculate_metrics(distorted, ground_truth)
            
            # Metrics AFTER denoising (denoised vs ground truth)
            metrics_after = calculate_metrics(denoised, ground_truth)
            
            result = {
                'filename': filename,
                'psnr_before': metrics_before['psnr'],
                'ssim_before': metrics_before['ssim'],
                'psnr_after': metrics_after['psnr'],
                'ssim_after': metrics_after['ssim'],
                'psnr_improvement': metrics_after['psnr'] - metrics_before['psnr'],
                'ssim_improvement': metrics_after['ssim'] - metrics_before['ssim']
            }
            all_metrics.append(result)
            
            # Save denoised image if requested
            if save_results:
                save_path = denoised_dir / filename
                save_image(denoised, str(save_path))
    
    # Compute averages
    avg_psnr_before = np.mean([m['psnr_before'] for m in all_metrics])
    avg_ssim_before = np.mean([m['ssim_before'] for m in all_metrics])
    avg_psnr_after = np.mean([m['psnr_after'] for m in all_metrics])
    avg_ssim_after = np.mean([m['ssim_after'] for m in all_metrics])
    avg_psnr_improvement = np.mean([m['psnr_improvement'] for m in all_metrics])
    avg_ssim_improvement = np.mean([m['ssim_improvement'] for m in all_metrics])
    
    summary = {
        'num_images': len(dataset),
        'average_psnr_before': float(avg_psnr_before),
        'average_ssim_before': float(avg_ssim_before),
        'average_psnr_after': float(avg_psnr_after),
        'average_ssim_after': float(avg_ssim_after),
        'average_psnr_improvement': float(avg_psnr_improvement),
        'average_ssim_improvement': float(avg_ssim_improvement),
        'per_image_results': all_metrics
    }
    
    return summary


def print_results(results: dict):
    """Pretty print evaluation results"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of images: {results['num_images']}")
    print()
    print("BEFORE Denoising (Distorted vs Ground Truth):")
    print(f"  Average PSNR: {results['average_psnr_before']:.2f} dB")
    print(f"  Average SSIM: {results['average_ssim_before']:.4f}")
    print()
    print("AFTER Denoising (Denoised vs Ground Truth):")
    print(f"  Average PSNR: {results['average_psnr_after']:.2f} dB")
    print(f"  Average SSIM: {results['average_ssim_after']:.4f}")
    print()
    print("IMPROVEMENT:")
    print(f"  PSNR Improvement: {results['average_psnr_improvement']:+.2f} dB")
    print(f"  SSIM Improvement: {results['average_ssim_improvement']:+.4f}")
    print("=" * 60)
    
    # Per-image results
    print("\nPer-Image Results:")
    print("-" * 80)
    print(f"{'Filename':<30} {'PSNR Before':>12} {'PSNR After':>12} {'Improvement':>12}")
    print("-" * 80)
    for r in results['per_image_results']:
        print(f"{r['filename']:<30} {r['psnr_before']:>12.2f} {r['psnr_after']:>12.2f} {r['psnr_improvement']:>+12.2f}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DnCNN model on test dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pre-trained model (auto-downloads)
  python evaluate_pretrained.py --data_dir data/test --noise_level 25
  
  # Use custom model
  python evaluate_pretrained.py --data_dir data/test --model_path checkpoints/best.pth
  
  # Save denoised images
  python evaluate_pretrained.py --data_dir data/test --save_results
  
  # List available pre-trained models
  python evaluate_pretrained.py --list_models
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='data/test',
                        help='Directory with distorted/ and ground_truth/ subdirs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (if not using pre-trained)')
    parser.add_argument('--noise_level', type=int, default=25, choices=[15, 25, 50],
                        help='Noise level for pre-trained grayscale model (15, 25, or 50)')
    parser.add_argument('--blind', action='store_true',
                        help='Use blind denoising model (recommended)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Process images as grayscale (default: color/RGB)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Specific model name (e.g., dncnn_color_blind)')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize all images to this size (default: 256). Use 0 for no resize.')
    parser.add_argument('--save_results', action='store_true',
                        help='Save denoised images')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--list_models', action='store_true',
                        help='List available pre-trained models and exit')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        list_available_models()
        return
    
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
    if args.model_path:
        # Load custom model
        print(f"Loading model from {args.model_path}...")
        in_channels = 1 if args.grayscale else 3
        model = load_pretrained_dncnn(args.model_path, device, in_channels)
    else:
        # Download and load pre-trained model
        if args.model_name:
            # Use explicitly specified model
            model_name = args.model_name
        elif args.grayscale:
            # Grayscale models
            if args.blind:
                model_name = 'dncnn_gray_blind'
            else:
                model_name = f'dncnn_{args.noise_level}'
        else:
            # COLOR images - MUST use color model!
            model_name = 'dncnn_color_blind'
            if not args.blind and args.noise_level != 25:
                print(f"  Note: dncnn_{args.noise_level} is grayscale-only.")
                print(f"  Using 'dncnn_color_blind' for color images instead.")
        
        print(f"Using pre-trained model: {model_name}")
        model_path = download_pretrained_model(model_name, save_dir='pretrained')
        
        in_channels = 1 if args.grayscale else 3
        model = load_pretrained_dncnn(model_path, device, in_channels)
    
    # Print model info
    info = get_model_info(model)
    print(f"Model: {info['total_parameters']:,} parameters, {info['num_layers']} layers")
    
    # Evaluate
    results = evaluate_on_dataset(
        model=model,
        data_dir=args.data_dir,
        device=device,
        save_results=args.save_results,
        results_dir=args.results_dir,
        grayscale=args.grayscale,
        resize=args.resize
    )
    
    # Print results
    print_results(results)
    
    # Save results to JSON
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_path / f"evaluation_{timestamp}.json"
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    main()

