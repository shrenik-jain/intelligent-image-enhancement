import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def evaluate_enhancement(original_low, original_high, enhanced, method_name):
    """ 
    Evaluate the enhancement by computing SSIM, PSNR, and MSE between:
    1. Original low-light image and original high-light image.
    2. Original low-light image and enhanced image.

    Arguments:
    original_low -- The original low-light image.
    original_high -- The original high-light image.
    enhanced -- The enhanced image.
    method_name -- Name of the enhancement method used.

    Returns:
    result -- Dictionary containing evaluation metrics.
    """
    result = {}
    # Convert to grayscale
    low_gray = cv2.cvtColor(original_low, cv2.COLOR_BGR2GRAY)
    high_gray = cv2.cvtColor(original_high, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # SSIM
    ssim_low_high, ssim_diff_low_high = ssim(low_gray, high_gray, full=True)
    ssim_low_enh, ssim_diff_low_enh = ssim(low_gray, enh_gray, full=True)
    # PSNR
    psnr_low_high = cv2.PSNR(low_gray, high_gray)
    psnr_low_enh = cv2.PSNR(low_gray, enh_gray)
    # MSE
    mse_low_high = np.mean((low_gray - high_gray) ** 2)
    mse_low_enh = np.mean((low_gray - enh_gray) ** 2)

    # Pretty prints
    print("**********************************************")
    print(f"Eval Results for {method_name}\n")
    print(f"SSIM  low vs high: {ssim_low_high:.4f}")
    print(f"SSIM  low vs enhanced : {ssim_low_enh:.4f}")
    print()
    print(f"PSNR  low vs high: {psnr_low_high:.2f} dB")
    print(f"PSNR  low vs enhanced : {psnr_low_enh:.2f} dB")
    print()
    print(f"MSE   low vs high: {mse_low_high:.4f}")
    print(f"MSE   low vs enhanced : {mse_low_enh:.4f}")
    print("**********************************************\n")

    # Store metrics in result dict
    result.update(
        {
            "ssim_low_high": ssim_low_high,
            "ssim_low_enh": ssim_low_enh,
            "psnr_low_high": psnr_low_high,
            "psnr_low_enh": psnr_low_enh,
            "mse_low_high": mse_low_high,
            "mse_low_enh": mse_low_enh,
        }
    )


def simple_enhancement(image_path, original_image_path, brightness=30, contrast=2):
    """
    A straightforward method to enhance low-light images by adjusting brightness 
    and contrast using OpenCV's addWeighted function.
    
    Arguments:
    image_path -- Path to the low-light image.
    original_image_path -- Path to the original high-light image.
    brightness -- Value to increase brightness.
    contrast -- Factor to increase contrast.
    
    Returns:
    enhanced_image -- The brightness and contrast enhanced image.
    results -- Dictionary containing evaluation metrics.
    """
    image = cv2.imread(image_path)
    original_high = cv2.imread(original_image_path)

    enhanced_image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    cv2.imwrite(SAVE_DIR + 'simple_enhancement.jpg', enhanced_image)
    results = evaluate_enhancement(image, original_high, enhanced_image, "Simple Enhancement")
    return enhanced_image, results


def clahe_enhancement(image_path, original_image_path, clip_limit=5.0, tile_grid_size=(4, 4)):
    """ 
    Global histogram equalization can increase the overall contrast of an image. 
    Contrast Limited Adaptive Histogram Equalization (CLAHE) is a more advanced 
    technique that enhances contrast locally, preventing over-enhancement in bright 
    areas and preserving details in dark regions.

    Arguments:
    image_path -- Path to the low-light image.
    original_image_path -- Path to the original high-light image.
    clip_limit -- Threshold for contrast limiting.
    tile_grid_size -- Size of grid for histogram equalization.

    Returns:
    enhanced_image -- The contrast-enhanced image.
    results -- Dictionary containing evaluation metrics.
    """
    image = cv2.imread(image_path)
    original_high = cv2.imread(original_image_path)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(SAVE_DIR + 'clahe_enhancement.jpg', enhanced_image)

    results = evaluate_enhancement(image, original_high, enhanced_image, "CLAHE Enhancement")
    return enhanced_image, results


def gamma_correction_enhancement(image_path, original_image_path, gamma=0.5):
    """
    Gamma correction is a non-linear operation used to encode and decode luminance 
    or tristimulus values in images. It can be used to adjust the brightness of an image.

    Arguments:
    image_path -- Path to the low-light image.
    original_image_path -- Path to the original high-light image.
    gamma -- Gamma value for correction.

    Returns:
    enhanced_image -- The gamma-corrected image.
    results -- Dictionary containing evaluation metrics.
    """
    image = cv2.imread(image_path)
    original_high = cv2.imread(original_image_path)

    enhanced_img = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    cv2.imwrite(SAVE_DIR + 'gamma_correction_enhancement.jpg', enhanced_img)

    results = evaluate_enhancement(image, original_high, enhanced_img, "Gamma Correction Enhancement")
    return enhanced_img, results


if __name__ == "__main__":
    SAVE_DIR = 'modified/'
    IMG_PATH = 'input/low_light_image.jpg'
    ORIGINAL_HIGH_PATH = 'input/original_high_light_image.jpg'
    
    _, _ = simple_enhancement(IMG_PATH, ORIGINAL_HIGH_PATH)
    _, _ = clahe_enhancement(IMG_PATH, ORIGINAL_HIGH_PATH)
    _, _ = gamma_correction_enhancement(IMG_PATH, ORIGINAL_HIGH_PATH)