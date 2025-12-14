######################
## Essential libraries
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

def kernel_psf(angle, d, size=20):
    """
    Create a Point Spread Function (PSF) kernel for motion blur.
    
    Args:
        angle: Motion blur angle in radians
        d: Distance/length of motion blur
        size: Size of the kernel (default: 20)
    
    Returns:
        PSF kernel as numpy array
    """
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    size2 = size // 2  # Division(floor)
    A[:,2] = (size2, size2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kernel = cv2.warpAffine(kernel, A, (size, size), flags=cv2.INTER_CUBIC)
    return kernel

def wiener_filter(img, kernel, K):
    """
    Apply Wiener filter for image deblurring.
    
    Args:
        img: Input image (single channel)
        kernel: PSF kernel
        K: Noise-to-signal ratio parameter
    
    Returns:
        Filtered image
    """
    kernel /= np.sum(kernel)
    copy_img = np.copy(img)
    copy_img = np.fft.fft2(copy_img)  # 2D fast fourier transform 
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)  # wiener formula implementation
    copy_img = copy_img * kernel  # conversion blurred to deblurred
    copy_img = np.abs(np.fft.ifft2(copy_img))  # 2D inverse fourier transform
    return copy_img

def process(ip_image, angle_deg=90, distance=20, K=0.0060, contrast=1.0, denoise=True, 
            denoise_strength=3, preserve_detail=True):
    """
    Process and deblur the input image using Wiener filter.
    
    Args:
        ip_image: Input blurred image (BGR format)
        angle_deg: Motion blur angle in degrees (default: 90)
        distance: Motion blur distance (default: 20)
        K: Noise-to-signal ratio for Wiener filter (default: 0.0060)
        contrast: Contrast adjustment factor (default: 1.0 - no adjustment)
        denoise: Whether to apply denoising to remove ringing artifacts (default: True)
        denoise_strength: Strength of denoising (1-10, default: 3)
        preserve_detail: Use gentler processing to preserve details (default: True)
    
    Returns:
        Deblurred image
    """
    ang = np.deg2rad(angle_deg)  # Convert angle to radians
    
    # Split into color channels
    b, g, r = cv2.split(ip_image)
    
    # Normalize split images 
    img_b = np.float32(b) / 255.0
    img_g = np.float32(g) / 255.0
    img_r = np.float32(r) / 255.0
    
    # PSF calculation 
    psf = kernel_psf(ang, distance)
    
    # Apply Wiener filter for all split images
    # Small value of K (SNR) - if 0, filter will become inverse filter
    filtered_img_b = wiener_filter(img_b, psf, K=K)
    filtered_img_g = wiener_filter(img_g, psf, K=K)
    filtered_img_r = wiener_filter(img_r, psf, K=K)
    
    # Merge to form colored image
    filtered_img = cv2.merge((filtered_img_b, filtered_img_g, filtered_img_r))
    
    # Convert float to uint8 with better normalization
    if preserve_detail:
        # Gentler normalization - preserve original brightness range
        filtered_img = np.clip(filtered_img * 255, 0, 255)
    else:
        # Original aggressive normalization
        filtered_img = np.clip(filtered_img * 255, 0, 255)
    
    filtered_img = np.uint8(filtered_img)
    
    # Optional contrast adjustment (default is now 1.0 = no change)
    if contrast != 1.0:
        filtered_img = cv2.convertScaleAbs(filtered_img, alpha=contrast)
    
    # Remove Gibbs phenomena or rings from the image
    if denoise:
        if preserve_detail:
            # Single pass with adjustable strength for less texture artifacts
            h = denoise_strength  # Luminance denoising strength
            hColor = denoise_strength  # Color denoising strength
            filtered_img = cv2.fastNlMeansDenoisingColored(
                filtered_img, None, h=h, hColor=hColor, 
                templateWindowSize=7, searchWindowSize=21
            )
        else:
            # Original double pass (more aggressive)
            filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15)
            filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15)
    
    return filtered_img

def load_image(image_path):
    """
    Load image from file, handling TIFF and other formats.
    Converts TIFF to BGR format compatible with OpenCV.
    
    Args:
        image_path: Path to input image
    
    Returns:
        Image in BGR format, or None if loading fails
    """
    # Get file extension
    _, ext = os.path.splitext(image_path.lower())
    
    # For TIFF files, try multiple methods
    if ext in ['.tiff', '.tif']:
        print(f"Detected TIFF format, attempting to load...")
        
        # Try tifffile first (best for scientific/microscopy TIFF)
        if TIFFFILE_AVAILABLE:
            try:
                print("Trying tifffile library...")
                frame = tifffile.imread(image_path)
                print(f"Loaded image with shape: {frame.shape}, dtype: {frame.dtype}")
                
                # Handle different data types - convert to uint8 first
                if frame.dtype == np.uint16:
                    print("Converting 16-bit image to 8-bit...")
                    frame = (frame / 256).astype(np.uint8)
                elif frame.dtype == np.float16:
                    print("Converting float16 image to 8-bit...")
                    # Convert to float32 first for better precision
                    frame = frame.astype(np.float32)
                    # Normalize to 0-255 range
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(frame, dtype=np.uint8)
                elif frame.dtype == np.float32 or frame.dtype == np.float64:
                    print("Converting float image to 8-bit...")
                    # Normalize to 0-255 range
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(frame, dtype=np.uint8)
                
                # Handle channel conversion - frame should be uint8 now
                if len(frame.shape) == 2:
                    # Grayscale, convert to BGR
                    print("Converting grayscale to BGR...")
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        # RGBA, convert to BGR
                        print("Converting RGBA to BGR...")
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3:
                        # RGB, convert to BGR
                        print("Converting RGB to BGR...")
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                return frame
                
            except Exception as e:
                print(f"Error loading TIFF with tifffile: {e}")
        
        # Try PIL
        try:
            print("Trying PIL library...")
            pil_img = Image.open(image_path)
            frame = np.array(pil_img)
            print(f"Loaded image with shape: {frame.shape}, dtype: {frame.dtype}")
            
            # Handle different data types
            if frame.dtype == np.uint16:
                print("Converting 16-bit image to 8-bit...")
                frame = (frame / 256).astype(np.uint8)
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
            print(f"Error loading TIFF with PIL: {e}")
            print("Trying with OpenCV...")
    
    # Try reading with OpenCV
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if frame is None:
        print(f"Error: Could not read image '{image_path}'")
        return None
    
    # Handle standard formats
    if len(frame.shape) == 2:
        # Grayscale image, convert to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        # RGBA image, convert to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Handle 16-bit images
    if frame.dtype == np.uint16:
        print("Converting 16-bit image to 8-bit...")
        frame = (frame / 256).astype(np.uint8)
    
    return frame

def main():
    """
    Main function to handle command-line interface.
    """
    parser = argparse.ArgumentParser(description='Motion deblur using Wiener filter (supports TIFF, JPG, PNG, etc.)')
    parser.add_argument('input_image', type=str, help='Path to input blurred image (supports TIFF, JPG, PNG)')
    parser.add_argument('output_image', type=str, help='Path to save deblurred image')
    parser.add_argument('--angle', type=float, default=90, help='Motion blur angle in degrees (default: 90)')
    parser.add_argument('--distance', type=int, default=20, help='Motion blur distance (default: 20)')
    parser.add_argument('--K', type=float, default=0.01, help='Noise-to-signal ratio for Wiener filter (default: 0.01, higher=smoother)')
    parser.add_argument('--contrast', type=float, default=1.0, help='Contrast adjustment factor (default: 1.0=no change)')
    parser.add_argument('--denoise-strength', type=int, default=3, help='Denoising strength 1-10 (default: 3)')
    parser.add_argument('--no-denoise', action='store_true', help='Skip denoising step')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive mode (may introduce artifacts)')
    parser.add_argument('--display', action='store_true', help='Display the result before saving')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found!")
        return
    
    # Read the input image with TIFF support
    print(f"Reading input image: {args.input_image}")
    frame = load_image(args.input_image)
    
    if frame is None:
        return
    
    print(f"Image shape: {frame.shape}")
    
    # Process the image
    print("Processing image with Wiener filter...")
    print(f"Parameters: angle={args.angle}°, distance={args.distance}, K={args.K}, contrast={args.contrast}")
    op_image = process(
        frame, 
        angle_deg=args.angle,
        distance=args.distance,
        K=args.K,
        contrast=args.contrast,
        denoise=not args.no_denoise,
        denoise_strength=args.denoise_strength,
        preserve_detail=not args.aggressive
    )
    
    # Display if requested
    if args.display:
        cv2.imshow("Original", frame)
        cv2.imshow("Deblurred", op_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the output image
    cv2.imwrite(args.output_image, op_image)
    print(f"Deblurred image saved to: {args.output_image}")

############################################################################################
## main function
############################################################################################

if __name__ == '__main__':
    main()

