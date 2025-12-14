"""
Evaluate deblurring results against ground truth using multiple metrics
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import argparse
import sys
import os

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

from PIL import Image

def load_image(image_path):
    """
    Load image from file, handling TIFF and other formats.
    """
    _, ext = os.path.splitext(image_path.lower())
    
    if ext in ['.tiff', '.tif']:
        if TIFFFILE_AVAILABLE:
            try:
                frame = tifffile.imread(image_path)
                
                # Handle different data types
                if frame.dtype == np.uint16:
                    frame = (frame / 256).astype(np.uint8)
                elif frame.dtype == np.float16:
                    frame = frame.astype(np.float32)
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(frame, dtype=np.uint8)
                elif frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(frame, dtype=np.uint8)
                
                # Handle channel conversion
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                return frame
            except Exception as e:
                print(f"Error loading TIFF with tifffile: {e}")
        
        try:
            pil_img = Image.open(image_path)
            frame = np.array(pil_img)
            
            if frame.dtype == np.uint16:
                frame = (frame / 256).astype(np.uint8)
            elif frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_min = frame.min()
                frame_max = frame.max()
                if frame_max > frame_min:
                    frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Error loading TIFF with PIL: {e}")
    
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if frame is None:
        return None
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    if frame.dtype == np.uint16:
        frame = (frame / 256).astype(np.uint8)
    
    return frame

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    return psnr(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index"""
    return ssim(img1, img2, channel_axis=2, data_range=255)

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error"""
    return mse(img1, img2)

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

def calculate_rmse(img1, img2):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(img1, img2))

def calculate_nrmse(img1, img2):
    """Calculate Normalized Root Mean Squared Error"""
    rmse_val = calculate_rmse(img1, img2)
    return rmse_val / (img1.max() - img1.min())

def calculate_edge_similarity(img1, img2):
    """Calculate edge similarity using Sobel edge detection"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate edges
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    edges1 = np.sqrt(sobelx1**2 + sobely1**2)
    
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    edges2 = np.sqrt(sobelx2**2 + sobely2**2)
    
    # Normalize
    edges1 = edges1 / (edges1.max() + 1e-8)
    edges2 = edges2 / (edges2.max() + 1e-8)
    
    # Calculate correlation
    correlation = np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1]
    return correlation

def calculate_sharpness(img):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def evaluate_images(ground_truth_path, *test_image_paths):
    """
    Evaluate test images against ground truth
    """
    print("="*80)
    print("IMAGE DEBLURRING EVALUATION")
    print("="*80)
    
    # Load ground truth
    print(f"\n✓ Ground truth: {ground_truth_path}")
    gt = load_image(ground_truth_path)
    if gt is None:
        print(f"Error: Could not load ground truth image!")
        return
    
    gt_sharpness = calculate_sharpness(gt)
    print(f"  Shape: {gt.shape}, Sharpness: {gt_sharpness:.2f}\n")
    
    results = []
    
    for test_path in test_image_paths:
        if not os.path.exists(test_path):
            print(f"\nWarning: {test_path} not found, skipping...")
            continue
            
        # Load test image
        test = load_image(test_path)
        if test is None:
            print(f"Error: Could not load {test_path}")
            continue
        
        # Resize if needed
        if gt.shape != test.shape:
            test = cv2.resize(test, (gt.shape[1], gt.shape[0]))
        
        # Calculate metrics
        psnr_val = calculate_psnr(gt, test)
        ssim_val = calculate_ssim(gt, test)
        mse_val = calculate_mse(gt, test)
        mae_val = calculate_mae(gt, test)
        rmse_val = calculate_rmse(gt, test)
        nrmse_val = calculate_nrmse(gt, test)
        edge_sim = calculate_edge_similarity(gt, test)
        sharpness = calculate_sharpness(test)
        sharpness_ratio = sharpness / gt_sharpness
        
        result = {
            'name': os.path.basename(test_path),
            'psnr': psnr_val,
            'ssim': ssim_val,
            'mse': mse_val,
            'mae': mae_val,
            'rmse': rmse_val,
            'nrmse': nrmse_val,
            'edge_similarity': edge_sim,
            'sharpness': sharpness,
            'sharpness_ratio': sharpness_ratio
        }
        results.append(result)
        
        print(f"✓ Processed: {os.path.basename(test_path)}")
    
    # Clean summary table
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("FINAL RESULTS - Each Method vs Ground Truth")
        print(f"{'='*80}\n")
        
        # Create clean table header
        print(f"{'Metric':<20}", end="")
        for r in results:
            # Shorten names for cleaner display
            name = r['name'].replace('.jpg', '').replace('mouse_', '')
            print(f"{name:>15}", end="")
        print()
        print("-" * 80)
        
        # PSNR row
        print(f"{'PSNR (dB)':<20}", end="")
        for r in results:
            print(f"{r['psnr']:>15.2f}", end="")
        print()
        
        # SSIM row
        print(f"{'SSIM':<20}", end="")
        for r in results:
            print(f"{r['ssim']:>15.4f}", end="")
        print()
        
        # MSE row
        print(f"{'MSE':<20}", end="")
        for r in results:
            print(f"{r['mse']:>15.2f}", end="")
        print()
        
        # Edge Similarity row
        print(f"{'Edge Similarity':<20}", end="")
        for r in results:
            print(f"{r['edge_similarity']:>15.4f}", end="")
        print()
        
        # Sharpness Ratio row
        print(f"{'Sharpness Ratio':<20}", end="")
        for r in results:
            print(f"{r['sharpness_ratio']:>15.2f}", end="")
        print()
        
        print("\n" + "="*80)
        print("Note: Higher is better for PSNR, SSIM, Edge Similarity")
        print("      Lower is better for MSE")
        print("      Sharpness Ratio close to 1.0 is ideal")
    
    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate deblurred images against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare blurred and deblurred images to ground truth
  python evaluate_deblur.py ground_truth.tiff blurred.tiff deblurred.jpg
  
  # Compare multiple deblurring methods
  python evaluate_deblur.py ground_truth.tiff blurred.tiff method1.jpg method2.jpg method3.jpg
        """
    )
    parser.add_argument('ground_truth', type=str, help='Path to ground truth image')
    parser.add_argument('test_images', type=str, nargs='+', help='Path(s) to test images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth image '{args.ground_truth}' not found!")
        sys.exit(1)
    
    evaluate_images(args.ground_truth, *args.test_images)

if __name__ == '__main__':
    main()

