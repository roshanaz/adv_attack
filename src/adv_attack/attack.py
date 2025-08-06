import torch
import torch.nn.functional as F
from typing import Optional
from .models import ResNet50Wrapper

def pgd_attack(
     model: ResNet50Wrapper,
    image: torch.Tensor,
    target_class: int,
    epsilon: float = 8/255,        # 8 pixel values in 0-255 scale
    alpha: float = 2/255,          # Step size 
    num_iterations: int = 7,       # Number of PGD iterations
    random_start: bool = True      
) -> torch.Tensor:
    """
    Generate adversarial example using PGD attack.
    
    The attack performs num_iterations with step size alpha, 
    while always staying within epsilon from the initial point.

    Reference: Madry et. al 2019, Towards Deep Learning Models Resistant to Adversarial Attacks
    value of epsilon = 8 pixels in barely noticeable to humans
    """
    original = model.preprocess(image)
    
    # Initialize adversarial image
    adversarial = original.clone()
    
    if random_start:
        # Add uniform random noise
        noise = torch.empty_like(adversarial).uniform_(-epsilon, epsilon)
        adversarial = adversarial + noise
        adversarial = torch.clamp(adversarial, 0, 1)
    
    target = torch.tensor([target_class], device=model.device)
    
    for iteration in range(num_iterations):
        adversarial = _pgd_step(model, adversarial, original, target, alpha, epsilon)
    
    return adversarial

def _pgd_step(
    model: ResNet50Wrapper,
    adversarial: torch.Tensor,
    original: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    epsilon: float
) -> torch.Tensor:
    """
    Single PGD iteration step.
    
    Args:
        model: ResNet50 model wrapper
        adversarial: Current adversarial image
        original: Original preprocessed image
        target: Target class tensor to enforce the network to predict
        alpha: Step size
        epsilon: Constraint on perturbation
        
    Returns:
        Updated adversarial image after one PGD step
    """
    # Enable gradients for adversarial image
    adversarial.requires_grad_(True)
    
    logits = model.get_logits_with_gradients(adversarial)
    
    loss = F.cross_entropy(logits, target)
    
    grad = torch.autograd.grad(loss, adversarial, retain_graph=False)[0]
    
    # Take step in negative gradient direction
    adversarial = adversarial - alpha * grad.sign()
    perturbation = adversarial - original
    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    adversarial = original + perturbation
    
    # clamp to valid image range
    adversarial = torch.clamp(adversarial, 0, 1)
    
    adversarial = adversarial.detach()
    
    return adversarial