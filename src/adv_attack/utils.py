import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Union, Tuple

def load_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Load image from path and convert to tensor.
    
    Args:
        image_path: Path to image file
        size: Target size (height, width) for ResNet50
        
    Returns:
        RGB image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor back to PIL Image for saving/display.
    
    Args:
        tensor: Image tensor
        
    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Clamp to valid range and convert
    tensor = torch.clamp(tensor, 0, 1)
    transform = transforms.ToPILImage()
    return transform(tensor.cpu())

def l_infinity_norm(perturbation: torch.Tensor) -> float:
    """
    Compute maximum pixel change.
    
    Args:
        perturbation: Difference between adversarial and original image
    """
    return torch.max(torch.abs(perturbation)).item()

def l2_norm(perturbation: torch.Tensor) -> float:
    """
    Compute Euclidean distance.
    
    Args:
        perturbation: Difference between adversarial and original image
    """
    return torch.norm(perturbation).item()

def compute_perturbation_metrics(original: torch.Tensor, adversarial: torch.Tensor) -> dict:
    """
    Compute comprehensive perturbation metrics.
    
    Args:
        original: Original image tensor
        adversarial: Adversarial image tensor
    """
    perturbation = adversarial - original
    
    return {
        'l_infinity': l_infinity_norm(perturbation),
        'l2': l2_norm(perturbation),
        'mean_abs_change': torch.mean(torch.abs(perturbation)).item(),
        'max_pixel_value': torch.max(adversarial).item(),
        'min_pixel_value': torch.min(adversarial).item()
    }
