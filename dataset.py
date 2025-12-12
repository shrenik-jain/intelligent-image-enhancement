"""
Dataset Classes for DnCNN Training and Evaluation

Provides loaders for paired distorted/ground-truth images
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Default resize dimensions
DEFAULT_RESIZE = (256, 256)


def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension"""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def parse_resize(resize: Union[None, int, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Parse resize parameter into (height, width) tuple
    
    Args:
        resize: None, int (square), or (height, width) tuple
        
    Returns:
        (height, width) tuple or None
    """
    if resize is None:
        return None
    elif isinstance(resize, int):
        return (resize, resize)
    elif isinstance(resize, (tuple, list)) and len(resize) == 2:
        return tuple(resize)
    else:
        raise ValueError(f"resize must be None, int, or (h, w) tuple, got {resize}")


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        img: Image as numpy array (H, W) or (H, W, C), values in [0, 1]
        target_size: (height, width) tuple
        
    Returns:
        Resized image as numpy array
    """
    h, w = target_size
    
    # Convert to PIL for resizing
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize((w, h), Image.LANCZOS)  # PIL uses (width, height)
    
    return np.array(pil_img, dtype=np.float32) / 255.0


def get_image_pairs(distorted_dir: str, ground_truth_dir: str) -> List[Tuple[str, str]]:
    """
    Get matching pairs of distorted and ground truth images
    
    Args:
        distorted_dir: Path to distorted images folder
        ground_truth_dir: Path to ground truth images folder
        
    Returns:
        List of (distorted_path, ground_truth_path) tuples
    """
    distorted_dir = Path(distorted_dir)
    ground_truth_dir = Path(ground_truth_dir)
    
    # Get all image files in distorted directory
    distorted_files = {f.stem: f for f in distorted_dir.iterdir() if is_image_file(f.name)}
    
    # Get all image files in ground truth directory
    gt_files = {f.stem: f for f in ground_truth_dir.iterdir() if is_image_file(f.name)}
    
    # Find matching pairs (by stem name)
    pairs = []
    for stem in distorted_files:
        if stem in gt_files:
            pairs.append((str(distorted_files[stem]), str(gt_files[stem])))
        else:
            print(f"Warning: No ground truth found for {distorted_files[stem].name}")
    
    if not pairs:
        raise ValueError(f"No matching image pairs found between {distorted_dir} and {ground_truth_dir}")
    
    print(f"Found {len(pairs)} matching image pairs")
    return pairs


class PairedImageDataset(Dataset):
    """
    Dataset for paired distorted/ground truth images
    
    Args:
        distorted_dir: Path to distorted images
        ground_truth_dir: Path to ground truth images
        resize: Resize images to this size. Can be:
                - None: no resizing
                - int: resize to square (e.g., 256 -> 256x256)
                - tuple: (height, width)
                Default: (256, 256)
        patch_size: If specified, extract random patches of this size (after resize)
        augment: Apply data augmentation (flip, rotate)
        grayscale: Convert to grayscale
    """
    
    def __init__(
        self,
        distorted_dir: str,
        ground_truth_dir: str,
        resize: Union[None, int, Tuple[int, int]] = DEFAULT_RESIZE,
        patch_size: Optional[int] = None,
        augment: bool = False,
        grayscale: bool = False
    ):
        self.pairs = get_image_pairs(distorted_dir, ground_truth_dir)
        self.resize = parse_resize(resize)
        self.patch_size = patch_size
        self.augment = augment
        self.grayscale = grayscale
        
        if self.resize:
            print(f"Images will be resized to {self.resize[0]}x{self.resize[1]}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image and convert to float32 [0, 1]"""
        img = Image.open(path)
        
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        return np.array(img, dtype=np.float32) / 255.0
    
    def _random_crop(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random patch from both images at same location"""
        h, w = img1.shape[:2]
        
        if h < self.patch_size or w < self.patch_size:
            # Resize if image is smaller than patch size
            scale = max(self.patch_size / h, self.patch_size / w) + 0.1
            new_h, new_w = int(h * scale), int(w * scale)
            img1 = resize_image(img1, (new_h, new_w))
            img2 = resize_image(img2, (new_h, new_w))
            h, w = new_h, new_w
        
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        if len(img1.shape) == 2:
            return (
                img1[top:top + self.patch_size, left:left + self.patch_size],
                img2[top:top + self.patch_size, left:left + self.patch_size]
            )
        return (
            img1[top:top + self.patch_size, left:left + self.patch_size, :],
            img2[top:top + self.patch_size, left:left + self.patch_size, :]
        )
    
    def _augment(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply same augmentation to both images"""
        # Random horizontal flip
        if random.random() > 0.5:
            img1 = np.fliplr(img1).copy()
            img2 = np.fliplr(img2).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            img1 = np.flipud(img1).copy()
            img2 = np.flipud(img2).copy()
        
        # Random 90 degree rotation
        k = random.randint(0, 3)
        if k > 0:
            img1 = np.rot90(img1, k).copy()
            img2 = np.rot90(img2, k).copy()
        
        return img1, img2
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to PyTorch tensor (C, H, W)"""
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]  # Add channel dimension
        else:
            img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img.copy()).float()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a pair of images
        
        Returns:
            distorted: Distorted image tensor (C, H, W)
            ground_truth: Ground truth image tensor (C, H, W)
            filename: Original filename
        """
        distorted_path, gt_path = self.pairs[idx]
        
        # Load images
        distorted = self._load_image(distorted_path)
        ground_truth = self._load_image(gt_path)
        
        # Resize both images to target size
        if self.resize is not None:
            distorted = resize_image(distorted, self.resize)
            ground_truth = resize_image(ground_truth, self.resize)
        else:
            # Ensure same size (resize ground truth to match distorted if different)
            if distorted.shape != ground_truth.shape:
                h, w = distorted.shape[:2]
                ground_truth = resize_image(ground_truth, (h, w))
        
        # Extract patches if specified (after resize)
        if self.patch_size is not None:
            distorted, ground_truth = self._random_crop(distorted, ground_truth)
        
        # Apply augmentation
        if self.augment:
            distorted, ground_truth = self._augment(distorted, ground_truth)
        
        # Convert to tensors
        distorted = self._to_tensor(distorted)
        ground_truth = self._to_tensor(ground_truth)
        
        filename = Path(distorted_path).name
        
        return distorted, ground_truth, filename


class SingleImageDataset(Dataset):
    """
    Dataset for single images (inference only, no ground truth)
    
    Args:
        image_dir: Path to images
        resize: Resize images to this size. Can be:
                - None: no resizing
                - int: resize to square (e.g., 256 -> 256x256)
                - tuple: (height, width)
                Default: (256, 256)
        grayscale: Convert to grayscale
    """
    
    def __init__(
        self, 
        image_dir: str, 
        resize: Union[None, int, Tuple[int, int]] = DEFAULT_RESIZE,
        grayscale: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.images = [f for f in self.image_dir.iterdir() if is_image_file(f.name)]
        self.resize = parse_resize(resize)
        self.grayscale = grayscale
        
        if not self.images:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.images)} images")
        if self.resize:
            print(f"Images will be resized to {self.resize[0]}x{self.resize[1]}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        """
        Get an image
        
        Returns:
            image: Image tensor (C, H, W)
            filename: Original filename
            original_size: (height, width) - original size before resize
        """
        path = self.images[idx]
        img = Image.open(path)
        original_size = img.size[::-1]  # (width, height) -> (height, width)
        
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Resize if specified
        if self.resize is not None:
            img = resize_image(img, self.resize)
        
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose(2, 0, 1)
        
        return torch.from_numpy(img).float(), path.name, original_size


def create_dataloaders(
    train_distorted_dir: str,
    train_gt_dir: str,
    test_distorted_dir: str,
    test_gt_dir: str,
    batch_size: int = 8,
    resize: Union[None, int, Tuple[int, int]] = DEFAULT_RESIZE,
    patch_size: Optional[int] = None,
    num_workers: int = 4,
    grayscale: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders
    
    Args:
        train_distorted_dir: Path to training distorted images
        train_gt_dir: Path to training ground truth images
        test_distorted_dir: Path to test distorted images
        test_gt_dir: Path to test ground truth images
        batch_size: Batch size
        resize: Resize all images to this size (default: 256x256)
        patch_size: Size of patches to extract during training (None = use full resized image)
        num_workers: Number of data loading workers
        grayscale: Use grayscale images
        
    Returns:
        train_loader, test_loader
    """
    train_dataset = PairedImageDataset(
        train_distorted_dir,
        train_gt_dir,
        resize=resize,
        patch_size=patch_size,
        augment=True,
        grayscale=grayscale
    )
    
    test_dataset = PairedImageDataset(
        test_distorted_dir,
        test_gt_dir,
        resize=resize,
        patch_size=None,  # Use full resized images for testing
        augment=False,
        grayscale=grayscale
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Can use larger batch since all images are same size now
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test dataset creation
    print("Dataset module loaded successfully!")
    print(f"Supported image extensions: {IMAGE_EXTENSIONS}")
    print(f"Default resize: {DEFAULT_RESIZE}")
    
    # Test resize parsing
    print("\nTesting resize parsing:")
    print(f"  parse_resize(None) = {parse_resize(None)}")
    print(f"  parse_resize(256) = {parse_resize(256)}")
    print(f"  parse_resize((128, 256)) = {parse_resize((128, 256))}")
