import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Union, Tuple
import json
import os

def get_imagenet_class_name(class_id: int) -> str:
    """
    Get ImageNet class name from class ID.
    
    Args:
        class_id: ImageNet class ID (0-999)
        
    Returns:
        Human-readable class name
    """
    try:
        json_path = "imagenet_class_index.json"
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(__file__), "..", "..", "imagenet_class_index.json")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        class_info = data.get(str(class_id))
        if class_info:
            class_name = class_info[1].replace('_', ' ')
            return class_name
        else:
            return f"Class {class_id}"
            
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return f"Class {class_id}"

    
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

def visualize_perturbation(
    original: torch.Tensor,
    adversarial: torch.Tensor, 
    input_filename: str,
    output_filename: str,
    amplify_factor: float = 15.0
) -> torch.Tensor:
    """
    Visualize and save perturbation analysis with histogram.
    
    Args:
        original: Original image tensor [3, 224, 224] in range [0,1]
        adversarial: Adversarial image tensor [3, 224, 224] in range [0,1]
        input_filename: Original input filename (e.g., "cat.jpg")
        output_filename: Adversarial output filename (e.g., "adversarial_cat.jpg")
        amplify_factor: How much to amplify perturbation for visibility
        
    Returns:
        perturbation: The computed perturbation tensor
    """
    # Compute perturbation
    perturbation = adversarial - original
    
    # Generate filenames based on input/output
    base_name = os.path.splitext(output_filename)[0]
    analysis_path = f"{base_name}_perturbation_analysis.png"
    perturbation_only_path = f"{base_name}_perturbation_only.png"
    histogram_path = f"{base_name}_perturbation_histogram.png"
    
    # Create 2x3 subplot for comprehensive analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Convert tensors to numpy for matplotlib
    orig_np = tensor_to_numpy_image(original)
    adv_np = tensor_to_numpy_image(adversarial)
    
    # Amplified perturbation for visualization
    pert_amplified = perturbation * amplify_factor
    pert_amplified = torch.clamp(pert_amplified + 0.5, 0, 1)  # Center around 0.5
    pert_amplified_np = tensor_to_numpy_image(pert_amplified)
    
    # Raw perturbation (actual values)
    pert_raw = perturbation + 0.5  # Center around 0.5 for visualization
    pert_raw = torch.clamp(pert_raw, 0, 1)
    pert_raw_np = tensor_to_numpy_image(pert_raw)
    
    # Difference visualization (absolute values)
    pert_abs = torch.abs(perturbation) * amplify_factor * 2  # Extra amplification for abs values
    pert_abs = torch.clamp(pert_abs, 0, 1)
    pert_abs_np = tensor_to_numpy_image(pert_abs)
    
    # Plot images
    axes[0,0].imshow(orig_np)
    axes[0,0].set_title(f"Original Image\n({os.path.basename(input_filename)})")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(adv_np) 
    axes[0,1].set_title(f"Adversarial Image\n({os.path.basename(output_filename)})")
    axes[0,1].axis('off')
    
    axes[0,2].imshow(pert_raw_np)
    axes[0,2].set_title("Perturbation\n(Actual Scale)")
    axes[0,2].axis('off')
    
    axes[1,0].imshow(pert_amplified_np)
    axes[1,0].set_title(f"Perturbation\n(Amplified {amplify_factor}x)")
    axes[1,0].axis('off')
    
    axes[1,1].imshow(pert_abs_np)
    axes[1,1].set_title(f"Absolute Perturbation\n(Amplified {amplify_factor*2}x)")
    axes[1,1].axis('off')
    
    # Histogram of perturbation values
    pert_flat = perturbation.flatten().cpu().numpy()
    axes[1,2].hist(pert_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1,2].set_title("Perturbation Value Distribution")
    axes[1,2].set_xlabel("Perturbation Value")
    axes[1,2].set_ylabel("Frequency")
    axes[1,2].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Max: {torch.max(torch.abs(perturbation)):.6f}\n"
    stats_text += f"Mean: {torch.mean(torch.abs(perturbation)):.6f}\n"
    stats_text += f"Std: {torch.std(perturbation):.6f}\n"
    stats_text += f"L∞: {torch.max(torch.abs(perturbation)):.6f}\n"
    stats_text += f"L2: {torch.norm(perturbation):.6f}"
    
    axes[1,2].text(0.02, 0.98, stats_text, transform=axes[1,2].transAxes, 
                   verticalalignment='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save standalone perturbation visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(pert_amplified_np)
    plt.title(f"Perturbation Matrix (Amplified {amplify_factor}x)")
    plt.axis('off')
    plt.savefig(perturbation_only_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save standalone histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pert_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Perturbation Value Distribution")
    plt.xlabel("Perturbation Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for key statistics
    plt.axvline(0, color='red', linestyle='--', alpha=0.8, label='Zero (no change)')
    plt.axvline(torch.mean(perturbation).item(), color='green', linestyle='--', alpha=0.8, label='Mean')
    plt.axvline(-8/255, color='orange', linestyle='--', alpha=0.8, label='−ε bound')
    plt.axvline(8/255, color='orange', linestyle='--', alpha=0.8, label='+ε bound')
    plt.legend()
    
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved perturbation analysis to: {analysis_path}")
    print(f"Saved perturbation matrix to: {perturbation_only_path}")
    print(f"Saved perturbation histogram to: {histogram_path}")
    
    return perturbation

def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array for matplotlib visualization.
    
    Args:
        tensor: Image tensor [3, H, W] or [1, 3, H, W]
        
    Returns:
        numpy array: [H, W, 3] for matplotlib
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).cpu().numpy()
