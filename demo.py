import sys
from adv_attack import ResNet50Wrapper, pgd_attack, load_image, tensor_to_pil, compute_perturbation_metrics, get_imagenet_class_name
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from adv_attack import visualize_perturbation

def main():
    """
    Demonstrate adversarial attack with command line arguments.
    
    Usage: uv run examples/demo.py input_image.png target_class output_image.png
    """
    if len(sys.argv) != 4:
        print("Usage: uv run examples/demo.py <input_image> <target_class> <output_image>")
        print("Example: uv run examples/demo.py cat.jpg 285 adversarial_cat.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    target_class = int(sys.argv[2])
    output_path = sys.argv[3]
    
    print("Adversarial Attack Demo")
    print("======================")
    print(f"Input image: {input_path}")
    print(f"Target class: {target_class}")
    print(f"Output image: {output_path}")
    print()
    
    model = ResNet50Wrapper()
    
    try:
        image = load_image(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    original_class, original_class_name, original_confidence = model.classify(image)
    print(f"Original prediction: Class {original_class} ({original_class_name}) - confidence: {original_confidence:.3f}")
    
    target_name = get_imagenet_class_name(target_class)
    print(f"Target class: {target_class} ({target_name})")

    if not (0 <= target_class <= 999):
        print(f"Error: Target class {target_class} must be between 0-999 for ImageNet")
        sys.exit(1)
    
    if target_class == original_class:
        print(f"Warning: Target class {target_class} is same as original prediction")
    
    adversarial_image = pgd_attack(
        model=model,
        image=image,
        target_class=target_class,
        epsilon=8/255,      # Standard perturbation bound
        alpha=2/255,        # Step size
        num_iterations=7,   
        random_start=True   
    )
    
    adversarial_class, adversarial_class_name, adv_confidence = model.classify(adversarial_image)
    print(f"Adversarial prediction: Class {adversarial_class} ({adversarial_class_name}) - confidence: {adv_confidence:.3f}")
    
    if adversarial_class == target_class:
        print("Attack succeeded! Model fooled into predicting target class.")
    else:
        print(f"Attack failed. Predicted {adversarial_class} instead of {target_class}.")
    
    # Compute perturbation metrics
    metrics = compute_perturbation_metrics(
        image, 
        adversarial_image
    )
    
    print("Perturbation Metrics:")
    print(f"l_inf norm: {metrics['l_infinity']:.6f}")
    print(f"L2 norm: {metrics['l2']:.6f}")
    print(f"Mean absolute change: {metrics['mean_abs_change']:.6f}")
    
    l_inf_pixels = metrics['l_infinity'] * 255
    print(f"Max pixel change: {l_inf_pixels:.1f}/255 (in original scale)")
    
    if metrics['l_infinity'] <= 8/255:
        print("Perturbation within imperceptible range")
    else:
        print("Perturbation may be visible to humans")
    
    # Generate perturbation visualization
    print("\nGenerating perturbation visualization...")
    perturbation = visualize_perturbation(
        original=image,
        adversarial=adversarial_image,
        input_filename=input_path,
        output_filename=output_path,
        amplify_factor=15.0
    )

    # Additional perturbation statistics
    print(f"\nDetailed Perturbation Analysis:")
    print(f"Perturbation range: [{torch.min(perturbation):.6f}, {torch.max(perturbation):.6f}]")
    print(f"Pixels with zero change: {(perturbation == 0).sum().item()} / {perturbation.numel()}")
    print(f"Pixels at epsilon bound: {(torch.abs(perturbation) >= (8/255 - 1e-6)).sum().item()}")
    
    try:
        adversarial_pil = tensor_to_pil(adversarial_image)
        adversarial_pil.save(output_path)
        print(f"\nSaved adversarial image to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()
