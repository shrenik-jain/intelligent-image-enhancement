######################
## Enhanced Motion Deblurring using Richardson-Lucy
## Better detail preservation than Wiener filter
######################

import cv2
import numpy as np
import os
import argparse
from PIL import Image
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

def kernel_psf(angle, d, size=65):
    """
    Create a Point Spread Function (PSF) kernel for motion blur.
    
    Args:
        angle: Motion blur angle in radians
        d: Distance/length of motion blur
        size: Size of the kernel (default: 65, larger for better accuracy)
    
    Returns:
        PSF kernel as numpy array
    """
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    size2 = size // 2
    A[:,2] = (size2, size2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kernel = cv2.warpAffine(kernel, A, (size, size), flags=cv2.INTER_CUBIC)
    return kernel

def richardson_lucy_deconvolution(image, psf, iterations=30, clip=True):
    """
    Richardson-Lucy deconvolution algorithm.
    Better preserves edges and details compared to Wiener filter.
    
    Args:
        image: Input blurred image (single channel, float)
        psf: Point spread function kernel
        iterations: Number of iterations (default: 30)
        clip: Clip result to valid range (default: True)
    
    Returns:
        Deconvolved image
    """
    # Ensure proper normalization
    psf = psf / np.sum(psf)
    psf_mirror = np.flip(psf)
    
    # Initial estimate
    estimate = np.copy(image)
    
    for i in range(iterations):
        # Convolve estimate with PSF
        conv = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
        
        # Avoid division by zero
        conv = np.maximum(conv, 1e-12)
        
        # Calculate relative blur
        relative_blur = image / conv
        
        # Convolve with flipped PSF
        correction = cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REFLECT)
        
        # Update estimate
        estimate = estimate * correction
        
        # Ensure non-negative values
        estimate = np.maximum(estimate, 0)
    
    if clip:
        estimate = np.clip(estimate, 0, 1)
    
    return estimate

def process_rl(ip_image, angle_deg=90, distance=20, iterations=30, 
               denoise_strength=2, bilateral_filter=True):
    """
    Process and deblur the input image using Richardson-Lucy deconvolution.
    
    Args:
        ip_image: Input blurred image (BGR format)
        angle_deg: Motion blur angle in degrees (default: 90)
        distance: Motion blur distance (default: 20)
        iterations: Number of R-L iterations (default: 30, higher=sharper but may amplify noise)
        denoise_strength: Strength of denoising (default: 2)
        bilateral_filter: Use bilateral filter for edge-preserving smoothing (default: True)
    
    Returns:
        Deblurred image
    """
    ang = np.deg2rad(angle_deg)
    
    # Split into color channels
    b, g, r = cv2.split(ip_image)
    
    # Normalize split images
    img_b = np.float32(b) / 255.0
    img_g = np.float32(g) / 255.0
    img_r = np.float32(r) / 255.0
    
    # PSF calculation
    psf = kernel_psf(ang, distance)
    
    print(f"Applying Richardson-Lucy deconvolution ({iterations} iterations)...")
    
    # Apply Richardson-Lucy for all channels
    deblurred_b = richardson_lucy_deconvolution(img_b, psf, iterations=iterations)
    deblurred_g = richardson_lucy_deconvolution(img_g, psf, iterations=iterations)
    deblurred_r = richardson_lucy_deconvolution(img_r, psf, iterations=iterations)
    
    # Merge channels
    deblurred = cv2.merge((deblurred_b, deblurred_g, deblurred_r))
    
    # Convert to uint8
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
    
    # Optional edge-preserving denoising
    if bilateral_filter:
        print("Applying bilateral filter for edge preservation...")
        deblurred = cv2.bilateralFilter(deblurred, d=9, sigmaColor=75, sigmaSpace=75)
    
    if denoise_strength > 0:
        print(f"Applying denoising (strength={denoise_strength})...")
        deblurred = cv2.fastNlMeansDenoisingColored(
            deblurred, None, 
            h=denoise_strength, 
            hColor=denoise_strength,
            templateWindowSize=7, 
            searchWindowSize=21
        )
    
    return deblurred

def load_image(image_path):
    """
    Load image from file, handling TIFF and other formats.
    """
    _, ext = os.path.splitext(image_path.lower())
    
    if ext in ['.tiff', '.tif']:
        print(f"Detected TIFF format, attempting to load...")
        
        if TIFFFILE_AVAILABLE:
            try:
                print("Trying tifffile library...")
                frame = tifffile.imread(image_path)
                print(f"Loaded image with shape: {frame.shape}, dtype: {frame.dtype}")
                
                # Handle different data types
                if frame.dtype == np.uint16:
                    print("Converting 16-bit image to 8-bit...")
                    frame = (frame / 256).astype(np.uint8)
                elif frame.dtype == np.float16:
                    print("Converting float16 image to 8-bit...")
                    frame = frame.astype(np.float32)
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(frame, dtype=np.uint8)
                elif frame.dtype == np.float32 or frame.dtype == np.float64:
                    print("Converting float image to 8-bit...")
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(frame, dtype=np.uint8)
                
                # Handle channel conversion
                if len(frame.shape) == 2:
                    print("Converting grayscale to BGR...")
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        print("Converting RGBA to BGR...")
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:
                        print("Converting RGB to BGR...")
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                return frame
                
            except Exception as e:
                print(f"Error loading TIFF with tifffile: {e}")
        
        try:
            print("Trying PIL library...")
            pil_img = Image.open(image_path)
            frame = np.array(pil_img)
            print(f"Loaded image with shape: {frame.shape}, dtype: {frame.dtype}")
            
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
            print("Trying with OpenCV...")
    
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if frame is None:
        print(f"Error: Could not read image '{image_path}'")
        return None
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    if frame.dtype == np.uint16:
        print("Converting 16-bit image to 8-bit...")
        frame = (frame / 256).astype(np.uint8)
    
    return frame

def main():
    """
    Main function to handle command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Motion deblur using Richardson-Lucy deconvolution (better detail preservation)'
    )
    parser.add_argument('input_image', type=str, help='Path to input blurred image')
    parser.add_argument('output_image', type=str, help='Path to save deblurred image')
    parser.add_argument('--angle', type=float, default=90, help='Motion blur angle in degrees (default: 90)')
    parser.add_argument('--distance', type=int, default=20, help='Motion blur distance (default: 20)')
    parser.add_argument('--iterations', type=int, default=30, help='R-L iterations (default: 30, higher=sharper)')
    parser.add_argument('--denoise', type=int, default=2, help='Denoising strength 0-10 (default: 2)')
    parser.add_argument('--no-bilateral', action='store_true', help='Skip bilateral filtering')
    parser.add_argument('--display', action='store_true', help='Display the result before saving')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found!")
        return
    
    print(f"Reading input image: {args.input_image}")
    frame = load_image(args.input_image)
    
    if frame is None:
        return
    
    print(f"Image shape: {frame.shape}")
    print(f"Parameters: angle={args.angle}°, distance={args.distance}, iterations={args.iterations}")
    
    # Process the image
    op_image = process_rl(
        frame,
        angle_deg=args.angle,
        distance=args.distance,
        iterations=args.iterations,
        denoise_strength=args.denoise,
        bilateral_filter=not args.no_bilateral
    )
    
    if args.display:
        cv2.imshow("Original", frame)
        cv2.imshow("Deblurred (Richardson-Lucy)", op_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cv2.imwrite(args.output_image, op_image)
    print(f"Deblurred image saved to: {args.output_image}")

if __name__ == '__main__':
    main()


