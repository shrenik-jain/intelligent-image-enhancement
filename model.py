"""
DnCNN Model Architecture (Modern PyTorch Implementation)

Based on the paper:
"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
by Kai Zhang et al. (TIP 2017)

This implementation is compatible with KAIR pre-trained weights.
https://github.com/cszn/KAIR
"""

import torch
import torch.nn as nn
import math
from collections import OrderedDict


class DnCNN(nn.Module):
    """
    DnCNN - Denoising Convolutional Neural Network
    
    Architecture:
    - First layer: Conv + ReLU
    - Middle layers: Conv + BatchNorm + ReLU
    - Last layer: Conv
    
    The network learns the residual (noise), and the clean image is obtained by:
    clean_image = noisy_image - predicted_noise
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        out_channels (int): Number of output channels (same as in_channels)
        num_layers (int): Total number of convolutional layers (17 for DnCNN-S, 20 for DnCNN-B)
        features (int): Number of feature maps in hidden layers (default: 64)
    """
    
    def __init__(self, in_channels=3, out_channels=3, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.features = features
        
        layers = []
        
        # First layer: Conv + ReLU (no BatchNorm)
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer: Conv only (output the residual/noise)
        layers.append(nn.Conv2d(features, out_channels, kernel_size=3, padding=1, bias=True))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming Normal initialization
        This is important for good training convergence
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Special initialization for BatchNorm as per original paper
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / 9.0 / 64.0))
                m.weight.data.clamp_(-0.025, 0.025)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        The network predicts the noise/residual, and we subtract it from the input
        to get the clean image.
        
        Args:
            x: Input noisy image (B, C, H, W)
            
        Returns:
            Denoised image (B, C, H, W)
        """
        # Predict the residual (noise)
        residual = self.model(x)
        # Subtract residual from input to get clean image
        return x - residual


def analyze_checkpoint(state_dict):
    """
    Analyze a checkpoint to determine the model structure
    
    Returns:
        dict with model configuration
    """
    # Determine the key prefix (could be 'model.' or nothing)
    sample_key = list(state_dict.keys())[0]
    prefix = 'model.' if sample_key.startswith('model.') else ''
    
    # Find all conv layer indices by looking for .weight with 4D shape
    conv_indices = []
    bn_indices = []
    
    for key, value in state_dict.items():
        # Remove prefix for parsing
        key_no_prefix = key[len(prefix):] if key.startswith(prefix) else key
        parts = key_no_prefix.split('.')
        
        if len(parts) >= 2 and parts[1] == 'weight' and value.dim() == 4:
            # This is a conv layer
            idx = int(parts[0])
            conv_indices.append(idx)
        elif '.running_mean' in key:
            # This is a BatchNorm layer
            idx = int(parts[0])
            bn_indices.append(idx)
    
    conv_indices = sorted(set(conv_indices))
    bn_indices = sorted(set(bn_indices))
    
    # Get input/output channels from first and last conv
    first_conv_key = f'{prefix}{conv_indices[0]}.weight'
    last_conv_key = f'{prefix}{conv_indices[-1]}.weight'
    
    first_shape = state_dict[first_conv_key].shape
    last_shape = state_dict[last_conv_key].shape
    
    in_channels = first_shape[1]
    out_channels = last_shape[0]
    features = first_shape[0]
    num_conv_layers = len(conv_indices)
    has_batchnorm = len(bn_indices) > 0
    
    return {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'features': features,
        'num_conv_layers': num_conv_layers,
        'has_batchnorm': has_batchnorm,
        'conv_indices': conv_indices,
        'bn_indices': bn_indices
    }


def build_model_from_checkpoint(state_dict):
    """
    Build a model that matches the checkpoint structure exactly
    
    Args:
        state_dict: The loaded state dictionary
        
    Returns:
        A model that can load the state_dict
    """
    config = analyze_checkpoint(state_dict)
    
    print(f"  Detected: {config['num_conv_layers']} conv layers, "
          f"in={config['in_channels']}, out={config['out_channels']}, "
          f"features={config['features']}, BatchNorm={config['has_batchnorm']}")
    
    # Determine the key prefix
    sample_key = list(state_dict.keys())[0]
    prefix = 'model.' if sample_key.startswith('model.') else ''
    
    # Build model with exact same structure
    model = nn.Sequential()
    
    conv_indices = config['conv_indices']
    bn_indices = set(config['bn_indices'])
    
    # Track what index we're at
    current_idx = 0
    
    for i, conv_idx in enumerate(conv_indices):
        # Add any ReLU layers between current position and this conv
        while current_idx < conv_idx:
            if current_idx not in bn_indices:
                # Could be ReLU (but we don't add it to Sequential with index)
                pass
            current_idx += 1
        
        # Determine conv parameters (use prefix for key lookup)
        weight_key = f'{prefix}{conv_idx}.weight'
        bias_key = f'{prefix}{conv_idx}.bias'
        
        weight_shape = state_dict[weight_key].shape
        has_bias = bias_key in state_dict
        
        out_ch, in_ch, kh, kw = weight_shape
        
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=kh//2, bias=has_bias)
        model.add_module(str(conv_idx), conv)
        current_idx = conv_idx + 1
        
        # Check if next index is BatchNorm
        if current_idx in bn_indices:
            bn = nn.BatchNorm2d(out_ch)
            model.add_module(str(current_idx), bn)
            current_idx += 1
        
        # Add ReLU after (except for last conv)
        if i < len(conv_indices) - 1:
            model.add_module(str(current_idx), nn.ReLU(inplace=True))
            current_idx += 1
    
    return model, config


class DnCNN_Flexible(nn.Module):
    """
    A flexible DnCNN wrapper that can load any KAIR checkpoint
    """
    
    def __init__(self, model_sequential, config):
        super().__init__()
        self.model = model_sequential
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.num_layers = config['num_conv_layers']
        self.features = config['features']
    
    def forward(self, x):
        residual = self.model(x)
        return x - residual


def load_pretrained_dncnn(model_path, device='cpu', in_channels=3, num_layers=17):
    """
    Load a pre-trained DnCNN model (supports both KAIR and our own trained models)
    
    This function automatically detects the checkpoint structure and builds
    a matching model.
    
    Args:
        model_path: Path to the .pth file
        device: 'cpu' or 'cuda' or 'mps'
        in_channels: Number of input channels (used for our trained models)
        num_layers: Number of layers (used for our trained models)
        
    Returns:
        Loaded model in eval mode
    """
    print(f"  Loading checkpoint from {model_path}...")
    
    # Always load to CPU first, then move to target device
    # This handles cases where model was saved on CUDA but we're loading on MPS or vice versa
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Determine the state dict and its source
    is_finetuned = False
    
    if 'model_state_dict' in checkpoint:
        # This is our own fine-tuned model checkpoint
        print(f"  Detected: Fine-tuned model checkpoint")
        state_dict = checkpoint['model_state_dict']
        is_finetuned = True
        
        # Check if it has 'model.' prefix (KAIR-style) or direct keys (our native DnCNN)
        sample_key = list(state_dict.keys())[0]
        
        if not sample_key.startswith('model.'):
            # It's our own native DnCNN structure (trained from scratch)
            print(f"  Structure: Native DnCNN")
            model = DnCNN(in_channels=in_channels, out_channels=in_channels, num_layers=17)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            print(f"  Model loaded successfully! (Fine-tuned native, {in_channels} channels)")
            return model
        else:
            print(f"  Structure: KAIR-compatible (fine-tuned from pre-trained)")
            # Continue to KAIR loading logic below
    else:
        # It's a KAIR pre-trained model or similar - extract state dict
        state_dict = checkpoint
        
        # Handle different state dict formats (some have nested structure)
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    state_dict = new_state_dict
    
    # Build model from checkpoint structure
    model_seq, config = build_model_from_checkpoint(state_dict)
    
    # Create wrapper
    model = DnCNN_Flexible(model_seq, config)
    
    # Remove 'model.' prefix from keys for loading into model.model
    # (checkpoint has 'model.0.weight', we need '0.weight' for the Sequential)
    load_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_key = k[6:]  # Remove 'model.' prefix
        else:
            new_key = k
        load_state_dict[new_key] = v
    
    # Load weights
    model.model.load_state_dict(load_state_dict, strict=True)
    model.to(device)
    model.eval()
    
    if is_finetuned:
        print(f"  Model loaded successfully! (Fine-tuned from KAIR)")
    else:
        print(f"  Model loaded successfully! (KAIR pre-trained)")
    
    return model


def create_model_for_training(in_channels=3, num_layers=17, pretrained_path=None, device='cpu'):
    """
    Create a DnCNN model for training/fine-tuning
    
    Args:
        in_channels: 1 for grayscale, 3 for color
        num_layers: Number of layers
        pretrained_path: Optional path to initialize from pre-trained weights
        device: Device to use
        
    Returns:
        Model ready for training
    """
    if pretrained_path:
        print(f"Loading pre-trained weights from {pretrained_path}...")
        try:
            model = load_pretrained_dncnn(pretrained_path, device)
            
            # Check if channels match
            if model.in_channels != in_channels:
                print(f"  Warning: Pre-trained model has {model.in_channels} channels, "
                      f"but requested {in_channels} channels.")
                print(f"  Creating new model and training from scratch.")
                model = DnCNN(in_channels=in_channels, out_channels=in_channels, 
                              num_layers=num_layers, features=64)
            else:
                print(f"  Successfully loaded pre-trained weights")
        except Exception as e:
            print(f"  Could not load pre-trained weights: {e}")
            print(f"  Training from scratch.")
            model = DnCNN(in_channels=in_channels, out_channels=in_channels, 
                          num_layers=num_layers, features=64)
    else:
        model = DnCNN(in_channels=in_channels, out_channels=in_channels, 
                      num_layers=num_layers, features=64)
    
    model.to(device)
    return model


class DnCNNColor(DnCNN):
    """DnCNN for color (RGB) images - convenience class"""
    def __init__(self, num_layers=17, features=64):
        super().__init__(in_channels=3, out_channels=3, num_layers=num_layers, features=features)


class DnCNNGray(DnCNN):
    """DnCNN for grayscale images - convenience class"""
    def __init__(self, num_layers=17, features=64):
        super().__init__(in_channels=1, out_channels=1, num_layers=num_layers, features=features)


def get_model_info(model):
    """Get model information (parameters, layers, etc.)"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'num_layers': getattr(model, 'num_layers', 'unknown'),
        'features': getattr(model, 'features', 'unknown'),
        'in_channels': getattr(model, 'in_channels', 'unknown'),
        'out_channels': getattr(model, 'out_channels', 'unknown')
    }


if __name__ == '__main__':
    # Quick test
    print("Testing DnCNN models...")
    
    # Test our model
    print("\n1. Testing custom DnCNN (color)...")
    model = DnCNN(in_channels=3, out_channels=3, num_layers=17)
    info = get_model_info(model)
    print(f"   Parameters: {info['total_parameters']:,}")
    
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"   Forward pass: {x.shape} -> {y.shape} ✓")
    
    # Test grayscale model
    print("\n2. Testing custom DnCNN (grayscale)...")
    model_gray = DnCNN(in_channels=1, out_channels=1, num_layers=17)
    info = get_model_info(model_gray)
    print(f"   Parameters: {info['total_parameters']:,}")
    
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = model_gray(x)
    print(f"   Forward pass: {x.shape} -> {y.shape} ✓")
    
    print("\nAll tests passed!")
