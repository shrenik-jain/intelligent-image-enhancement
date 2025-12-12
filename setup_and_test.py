#!/usr/bin/env python3
"""
Setup and Test Script for DnCNN Training Environment

This script:
1. Checks if all dependencies are installed
2. Downloads a pre-trained model
3. Runs a quick test to verify everything works
4. Creates sample data structure

Run this first to verify your setup!

Usage:
    python setup_and_test.py
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'scipy': 'SciPy',
        'tqdm': 'tqdm'
    }
    
    missing = []
    installed = []
    
    for module, name in required.items():
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
            installed.append(f"  ✓ {name}: {version}")
        except ImportError:
            missing.append(name)
    
    for msg in installed:
        print(msg)
    
    if missing:
        print(f"\n  ✗ Missing packages: {', '.join(missing)}")
        print(f"\n  Install with: pip install -r requirements_training.txt")
        return False
    
    print("\n  All dependencies installed!")
    return True


def check_gpu():
    """Check GPU availability (CUDA or MPS)"""
    print("\n" + "=" * 60)
    print("CHECKING GPU (CUDA/MPS)")
    print("=" * 60)
    
    try:
        import torch
        
        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            return True
        
        # Check MPS (Apple Metal)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✓ MPS (Apple Metal) available")
            print(f"  ✓ PyTorch version: {torch.__version__}")
            return True
        
        print("  ⚠ No GPU available (CUDA/MPS) - will use CPU (slower)")
        return False
    except Exception as e:
        print(f"  ⚠ Error checking GPU: {e}")
        return False


def test_model():
    """Test model creation and forward pass"""
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    try:
        import torch
        from model import DnCNN, get_model_info
        
        # Create model
        model = DnCNN(in_channels=3, out_channels=3, num_layers=17)
        info = get_model_info(model)
        print(f"  ✓ Model created: {info['total_parameters']:,} parameters")
        
        # Test forward pass
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        
        if y.shape == x.shape:
            print(f"  ✓ Forward pass successful: {x.shape} -> {y.shape}")
        else:
            print(f"  ✗ Shape mismatch: {x.shape} -> {y.shape}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def download_pretrained():
    """Download a pre-trained model"""
    print("\n" + "=" * 60)
    print("DOWNLOADING PRE-TRAINED MODEL")
    print("=" * 60)
    
    try:
        from utils import download_pretrained_model, list_available_models
        
        # List available models
        list_available_models()
        
        # Download default model
        print("\nDownloading dncnn_25 (default)...")
        model_path = download_pretrained_model('dncnn_25', save_dir='pretrained')
        print(f"  ✓ Downloaded to: {model_path}")
        
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("    (This requires internet connection)")
        return False


def check_data_structure():
    """Check if data directories exist"""
    print("\n" + "=" * 60)
    print("CHECKING DATA STRUCTURE")
    print("=" * 60)
    
    required_dirs = [
        'data/train/distorted',
        'data/train/ground_truth',
        'data/test/distorted',
        'data/test/ground_truth'
    ]
    
    base = Path(__file__).parent
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base / dir_path
        if full_path.exists():
            # Count images
            images = list(full_path.glob('*.jpg')) + list(full_path.glob('*.png')) + list(full_path.glob('*.jpeg'))
            print(f"  ✓ {dir_path}: {len(images)} images")
        else:
            print(f"  ⚠ {dir_path}: Directory missing (created)")
            full_path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    if not all_exist:
        print("\n  ⚠ Data directories created. Please add your images!")
        print("    See PLAN.md for the required data structure.")
    
    return True


def create_sample_data():
    """Create sample synthetic data for testing"""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE DATA (for testing)")
    print("=" * 60)
    
    try:
        import numpy as np
        from PIL import Image
        
        base = Path(__file__).parent
        
        # Check if sample data already exists
        sample_gt = base / 'data/test/ground_truth/sample_001.png'
        if sample_gt.exists():
            print("  ✓ Sample data already exists")
            return True
        
        # Create sample images
        print("  Creating 5 sample image pairs...")
        
        for i in range(5):
            # Create a simple gradient/pattern image as ground truth
            size = (256, 256, 3)
            gt = np.zeros(size, dtype=np.uint8)
            
            # Add some patterns
            for c in range(3):
                gt[:, :, c] = np.clip(
                    np.outer(np.arange(256), np.ones(256)) * (c + 1) / 3 +
                    np.random.randint(0, 50, size[:2]),
                    0, 255
                ).astype(np.uint8)
            
            # Create noisy version
            noise = np.random.randn(*size) * 25  # Noise level 25
            noisy = np.clip(gt.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            # Save
            gt_path = base / f'data/test/ground_truth/sample_{i+1:03d}.png'
            noisy_path = base / f'data/test/distorted/sample_{i+1:03d}.png'
            
            Image.fromarray(gt).save(gt_path)
            Image.fromarray(noisy).save(noisy_path)
        
        print(f"  ✓ Created 5 sample pairs in data/test/")
        print("  Note: These are synthetic samples for testing only!")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to create sample data: {e}")
        return False


def test_evaluation():
    """Test the evaluation script on sample data"""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION PIPELINE")
    print("=" * 60)
    
    try:
        import torch
        from model import DnCNN
        from dataset import PairedImageDataset
        from utils import calculate_metrics
        
        base = Path(__file__).parent
        distorted_dir = base / 'data/test/distorted'
        gt_dir = base / 'data/test/ground_truth'
        
        # Check if we have sample data
        distorted_files = list(distorted_dir.glob('*.png')) + list(distorted_dir.glob('*.jpg'))
        if not distorted_files:
            print("  ⚠ No test data found. Run create_sample_data() first.")
            return False
        
        # Create dataset
        dataset = PairedImageDataset(str(distorted_dir), str(gt_dir), grayscale=False)
        print(f"  ✓ Dataset created: {len(dataset)} pairs")
        
        # Load one sample
        distorted, gt, filename = dataset[0]
        print(f"  ✓ Sample loaded: {filename}, shape: {distorted.shape}")
        
        # Test metrics
        metrics = calculate_metrics(distorted, gt)
        print(f"  ✓ Metrics (before denoising): PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")
        
        print("\n  Pipeline test successful!")
        return True
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("  DnCNN TRAINING ENVIRONMENT SETUP & TEST")
    print("=" * 60)
    
    results = {}
    
    # Run all checks
    results['dependencies'] = check_dependencies()
    if not results['dependencies']:
        print("\n⚠ Please install dependencies first!")
        sys.exit(1)
    
    results['gpu'] = check_gpu()
    results['model'] = test_model()
    results['data_structure'] = check_data_structure()
    results['sample_data'] = create_sample_data()
    results['evaluation'] = test_evaluation()
    
    # Try to download pretrained (requires internet)
    try:
        results['pretrained'] = download_pretrained()
    except:
        results['pretrained'] = False
        print("  ⚠ Could not download pretrained model (no internet?)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL" if name in ['dependencies', 'model'] else "⚠ WARN"
        if not passed and name in ['dependencies', 'model']:
            all_passed = False
        print(f"  {name:20s}: {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ Setup complete! You're ready to train.")
        print("\nNext steps:")
        print("  1. Add your data to data/train/ and data/test/")
        print("  2. Run: python evaluate_pretrained.py --data_dir data/test")
        print("  3. Run: python train.py --data_dir data/train --val_dir data/test")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

