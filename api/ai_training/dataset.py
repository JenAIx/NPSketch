"""
Dataset and DataLoader for Training Data Images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import io
from typing import List, Dict, Tuple
import json


class DrawingDataset(Dataset):
    """
    PyTorch Dataset for training data images.
    """
    
    def __init__(self, images_data: List[Dict], target_feature: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            images_data: List of dicts with 'processed_image_data', 'features_data'
            target_feature: Name of feature to predict (e.g., 'Total_Score')
            transform: Optional torchvision transforms
        """
        self.images_data = images_data
        self.target_feature = target_feature
        self.transform = transform
        
        # Filter: Only keep images that have the target feature
        self.valid_indices = []
        for idx, img_data in enumerate(images_data):
            features = {}
            if img_data.get('features_data'):
                try:
                    features = json.loads(img_data['features_data'])
                except:
                    pass
            
            if target_feature in features:
                self.valid_indices.append(idx)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        real_idx = self.valid_indices[idx]
        img_data = self.images_data[real_idx]
        
        # Load image from bytes
        image_bytes = img_data['processed_image_data']
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Get target value
        features = json.loads(img_data['features_data'])
        target_value = float(features[self.target_feature])
        target_tensor = torch.tensor([target_value], dtype=torch.float32)
        
        return img_tensor, target_tensor


def create_dataloaders(
    images_data: List[Dict],
    target_feature: str,
    train_split: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders.
    
    Args:
        images_data: List of image data dicts
        target_feature: Feature name to predict
        train_split: Fraction for training (rest for validation)
        batch_size: Batch size
        shuffle: Shuffle training data
    
    Returns:
        (train_loader, val_loader, stats)
    """
    # Create dataset
    dataset = DrawingDataset(images_data, target_feature)
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # No multiprocessing in Docker
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    stats = {
        "total_samples": total_size,
        "train_samples": train_size,
        "val_samples": val_size,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "batch_size": batch_size
    }
    
    return train_loader, val_loader, stats

