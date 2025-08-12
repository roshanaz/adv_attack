import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from adv_attack.utils import get_imagenet_class_name

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _get_device(device: str = 'auto') -> torch.device:
    """Get the appropriate device for computation."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)

class ResNet50Wrapper:
    """
    Simple wrapper for ResNet50 adversarial attacks.
    Provides classification and gradient-enabled logits for attacks.
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = _get_device(device)
        self.model = self._load_model()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
    def _load_model(self) -> nn.Module:
        """Load pre-trained ResNet50."""
        model = models.resnet50(pretrained=True)
        model.eval()
        model.to(self.device)
        return model
        
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for ResNet50.
        Expects RGB image in range [0, 1].
        """
        image = image.to(self.device)
        
        # Add batch dimension if single image
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        return self.normalize(image)
    
    def classify(self, image: torch.Tensor) -> tuple[int, str, float]:
        """
        Classify image and return predicted class ID, name, and confidence.
        
        Args:
            image: RGB image tensor in range [0, 1]
            
        Returns:
            tuple: (class_id, class_name, confidence)
                - class_id: Predicted class ID (0-999 for ImageNet)
                - class_name: Human-readable class name  
                - confidence: Prediction confidence (0.0 to 1.0)
        """
        preprocessed = self.preprocess(image)
        
        with torch.no_grad():  # No gradients needed for classification
            logits = self.model(preprocessed)
            probabilities = torch.softmax(logits, dim=1)
            confidence, class_id = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            class_id = class_id.item()
        
        class_name = get_imagenet_class_name(class_id)
        return class_id, class_name, confidence
    
    def get_logits_with_gradients(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get model logits with gradients enabled for adversarial attacks.
        
        Args:
            image: Preprocessed image tensor (already normalized)
            logits: Raw logits tensor with gradients
        """
        # Ensure gradients are enabled for the image
        image.requires_grad_(True)
        
        # Forward pass with gradients
        logits = self.model(image)
        
        return logits