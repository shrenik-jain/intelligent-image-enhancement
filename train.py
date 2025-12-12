#!/usr/bin/env python3
"""
Fine-tune DnCNN on Custom Dataset

This script fine-tunes a pre-trained DnCNN model on your training data.

Usage:
    # Basic fine-tuning
    python train.py --data_dir data/train --epochs 50
    
    # Fine-tuning with custom parameters
    python train.py --data_dir data/train --epochs 100 --batch_size 16 --lr 0.0001
    
    # Resume training from checkpoint
    python train.py --data_dir data/train --resume checkpoints/dncnn_finetuned_epoch_20.pth
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import DnCNN, load_pretrained_dncnn, get_model_info
from dataset import PairedImageDataset
from utils import (
    calculate_metrics, download_pretrained_model, AverageMeter,
    adjust_learning_rate
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> dict:
    """
    Train for one epoch
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for distorted, ground_truth, _ in pbar:
        # Move to device
        distorted = distorted.to(device)
        ground_truth = ground_truth.to(device)
        
        # Forward pass
        denoised = model(distorted)
        
        # Compute loss
        loss = criterion(denoised, ground_truth)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        denoised_clamped = torch.clamp(denoised, 0, 1)
        batch_metrics = calculate_metrics(denoised_clamped, ground_truth)
        
        # Update meters
        batch_size = distorted.size(0)
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(batch_metrics['psnr'], batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'psnr': f'{psnr_meter.avg:.2f}'
        })
    
    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """
    Validate the model
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    with torch.no_grad():
        for distorted, ground_truth, _ in tqdm(val_loader, desc='Validating', leave=False):
            distorted = distorted.to(device)
            ground_truth = ground_truth.to(device)
            
            # Forward pass
            denoised = model(distorted)
            denoised = torch.clamp(denoised, 0, 1)
            
            # Compute loss
            loss = criterion(denoised, ground_truth)
            
            # Compute metrics
            metrics = calculate_metrics(denoised, ground_truth)
            
            # Update meters
            loss_meter.update(loss.item(), distorted.size(0))
            psnr_meter.update(metrics['psnr'], distorted.size(0))
            ssim_meter.update(metrics['ssim'], distorted.size(0))
    
    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def plot_training_curves(history: list, save_dir: str):
    """
    Plot training and validation curves
    
    Args:
        history: List of dicts with training metrics per epoch
        save_dir: Directory to save plots
    """
    if not history:
        print("No training history to plot")
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_psnr = [h['train_psnr'] for h in history]
    
    # Extract validation data (only for epochs that have it)
    val_epochs = []
    val_loss = []
    val_psnr = []
    val_ssim = []
    
    for h in history:
        if 'val_loss' in h:
            val_epochs.append(h['epoch'])
            val_loss.append(h['val_loss'])
            val_psnr.append(h['val_psnr'])
            if 'val_ssim' in h:
                val_ssim.append(h['val_ssim'])
    
    has_val = len(val_epochs) > 0
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    if has_val:
        ax1.plot(val_epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([epochs[0], epochs[-1]])
    
    # Plot 2: PSNR curves
    ax2 = axes[1]
    ax2.plot(epochs, train_psnr, 'b-', label='Train PSNR', linewidth=2)
    if has_val:
        ax2.plot(val_epochs, val_psnr, 'r-', label='Validation PSNR', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Training & Validation PSNR', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([epochs[0], epochs[-1]])
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")
    
    # Also plot SSIM if available
    if has_val and len(val_ssim) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(val_epochs, val_ssim, 'g-', label='Validation SSIM', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('SSIM', fontsize=12)
        ax.set_title('Validation SSIM', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([epochs[0], epochs[-1]])
        
        ssim_path = Path(save_dir) / 'training_ssim.png'
        plt.savefig(ssim_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SSIM curve saved to {ssim_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune DnCNN on custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --data_dir data/train
  
  # Train with validation
  python train.py --data_dir data/train --val_dir data/test
  
  # Longer training with smaller learning rate
  python train.py --data_dir data/train --epochs 100 --lr 0.00005
  
  # Resume from checkpoint
  python train.py --data_dir data/train --resume checkpoints/last.pth
        """
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Training data directory (must have distorted/ and ground_truth/)')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data directory (optional)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Size of training patches (set to 0 to use full resized image)')
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize all images to this size (default: 256). Use 0 for no resize.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 regularization)')
    
    # Model arguments
    parser.add_argument('--noise_level', type=int, default=25,
                        help='Noise level for pre-trained model initialization')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch (no pre-trained weights)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--grayscale', action='store_true',
                        help='Train on grayscale images')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, or auto)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
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
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    train_distorted = os.path.join(args.data_dir, 'distorted')
    train_gt = os.path.join(args.data_dir, 'ground_truth')
    
    if not os.path.exists(train_distorted) or not os.path.exists(train_gt):
        print(f"Error: Training data directory must contain 'distorted' and 'ground_truth' subdirectories.")
        print(f"Expected:\n  {train_distorted}\n  {train_gt}")
        sys.exit(1)
    
    # Create training dataset with resize
    resize_size = args.resize if args.resize > 0 else None
    patch_size = args.patch_size if args.patch_size > 0 else None
    
    train_dataset = PairedImageDataset(
        train_distorted, train_gt,
        resize=resize_size,
        patch_size=patch_size,
        augment=True,
        grayscale=args.grayscale
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Training set: {len(train_dataset)} images")
    
    # Create validation dataset if specified
    val_loader = None
    if args.val_dir:
        val_distorted = os.path.join(args.val_dir, 'distorted')
        val_gt = os.path.join(args.val_dir, 'ground_truth')
        
        val_dataset = PairedImageDataset(
            val_distorted, val_gt,
            resize=resize_size,
            patch_size=None,
            augment=False,
            grayscale=args.grayscale
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        print(f"Validation set: {len(val_dataset)} images")
    
    # Create model
    in_channels = 1 if args.grayscale else 3
    
    if args.resume:
        # Resume from checkpoint
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Use the flexible loader to handle both KAIR and native model formats
        model = load_pretrained_dncnn(args.resume, device, in_channels)
        start_epoch = checkpoint['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch}")
        
    elif args.from_scratch:
        # Train from scratch
        print("Training from scratch...")
        model = DnCNN(in_channels=in_channels, out_channels=in_channels, num_layers=17)
        start_epoch = 1
        
    else:
        # Load pre-trained model
        # Select appropriate model based on color/grayscale
        if args.grayscale:
            model_name = f'dncnn_{args.noise_level}'
        else:
            model_name = 'dncnn_color_blind'  # Only color model available
            print(f"  Note: Using 'dncnn_color_blind' for color images")
        
        print(f"Loading pre-trained model: {model_name}")
        model_path = download_pretrained_model(model_name, save_dir='pretrained')
        model = load_pretrained_dncnn(model_path, device, in_channels)
        start_epoch = 1
    
    model = model.to(device)
    
    # Print model info
    info = get_model_info(model)
    print(f"Model: {info['total_parameters']:,} parameters")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state restored")
        except Exception as e:
            print(f"  Warning: Could not restore optimizer state: {e}")
            print("  Starting with fresh optimizer")
    
    # Training loop
    best_psnr = 0
    training_history = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate every 10 epochs (or on first/last epoch)
        should_validate = val_loader and (epoch % 10 == 0 or epoch == start_epoch or epoch == args.epochs)
        
        if should_validate:
            val_metrics = validate(model, val_loader, criterion, device)
            current_psnr = val_metrics['psnr']
        else:
            val_metrics = None
            current_psnr = train_metrics['psnr']
        
        # Log progress
        log_str = f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_metrics['loss']:.4f} | Train PSNR: {train_metrics['psnr']:.2f} dB"
        if val_metrics:
            log_str += f" | Val PSNR: {val_metrics['psnr']:.2f} dB | Val SSIM: {val_metrics['ssim']:.4f}"
        print(log_str)
        
        # Save history
        history_entry = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_psnr': train_metrics['psnr']
        }
        if val_metrics:
            history_entry.update({
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim']
            })
        training_history.append(history_entry)
        
        # Save best model (only update best if we validated)
        if should_validate and current_psnr > best_psnr:
            best_psnr = current_psnr
            save_checkpoint(
                model, optimizer, epoch,
                {'best_psnr': best_psnr},
                checkpoint_dir / 'dncnn_finetuned_best.pth'
            )
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                train_metrics,
                checkpoint_dir / f'dncnn_finetuned_epoch_{epoch}.pth'
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs,
        {'final_psnr': current_psnr},
        checkpoint_dir / 'dncnn_finetuned_final.pth'
    )
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(training_history, checkpoint_dir)
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training history saved to: {history_path}")
    print(f"Training curves saved to: {checkpoint_dir}/training_curves.png")
    print("=" * 60)


if __name__ == '__main__':
    main()

