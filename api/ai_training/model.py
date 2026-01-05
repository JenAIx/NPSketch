"""
CNN Model for Drawing Classification

Uses ResNet-18 architecture pre-trained on ImageNet,
fine-tuned for neuropsychological drawing assessment.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Dict, List


class DrawingClassifier(nn.Module):
    """
    CNN model for drawing classification/regression.
    
    Based on ResNet-18 with custom output head for multi-target prediction.
    """
    
    def __init__(self, num_outputs: int = 1, pretrained: bool = True, use_sigmoid: bool = False):
        """
        Initialize the model.
        
        Args:
            num_outputs: Number of output neurons (features to predict)
            pretrained: Use ImageNet pre-trained weights
            use_sigmoid: Use Sigmoid activation at output (for regression with normalized targets)
        """
        super(DrawingClassifier, self).__init__()
        self.use_sigmoid = use_sigmoid
        
        # Load pre-trained ResNet-18 (using modern API)
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Modify first conv layer to accept grayscale input (1 channel)
        # Convert RGB weights to grayscale by averaging
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        if pretrained:
            # Average RGB weights to grayscale
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Get number of features from final layer
        num_features = self.backbone.fc.in_features
        
        # Replace final layer with custom head
        if use_sigmoid:
            # For regression with normalized targets [0, 1]
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_outputs),
                nn.Sigmoid()  # Ensures output in [0, 1]
            )
        else:
            # For classification or raw regression
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_outputs)
            )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone weights (only train final layers)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze final layers
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model_summary(model: DrawingClassifier) -> Dict:
    """Get detailed model summary statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get output size
    output_size = model.backbone.fc[-1].out_features if isinstance(model.backbone.fc, nn.Sequential) else 1
    
    # Get head architecture details
    head_layers = []
    if isinstance(model.backbone.fc, nn.Sequential):
        for i, layer in enumerate(model.backbone.fc):
            if isinstance(layer, nn.Linear):
                head_layers.append({
                    "type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "parameters": layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                })
            elif isinstance(layer, nn.Dropout):
                head_layers.append({
                    "type": "Dropout",
                    "p": layer.p
                })
            elif isinstance(layer, nn.ReLU):
                head_layers.append({
                    "type": "ReLU"
                })
    
    return {
        "name": "DrawingClassifier",
        "architecture": "ResNet-18",
        "backbone": "ResNet-18 (ImageNet pre-trained)",
        "input_size": "568×274×1",
        "input_channels": 1,
        "output_neurons": output_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "head_layers": head_layers,
        "pretrained_weights": "ImageNet1K_V1",
        "framework": "PyTorch"
    }

