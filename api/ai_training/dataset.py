"""
Dataset and DataLoader for Training Data Images

Supports loading from:
1. Database (original implementation)
2. Augmented data directory (new)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import io
import cv2
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


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
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders with stratified split.
    
    Args:
        images_data: List of image data dicts
        target_feature: Feature name to predict
        train_split: Fraction for training (rest for validation)
        batch_size: Batch size
        shuffle: Shuffle training data
        random_seed: Random seed for reproducibility
    
    Returns:
        (train_loader, val_loader, stats)
    """
    # Create full dataset to get target values
    full_dataset = DrawingDataset(images_data, target_feature)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No samples with feature '{target_feature}'")
    
    # Extract all target values for stratified split
    all_targets = []
    for i in range(len(full_dataset)):
        _, target = full_dataset[i]
        all_targets.append(target.item())
    
    all_targets = np.array(all_targets)
    
    # Import split strategy
    try:
        from .split_strategy import get_split_recommendation, stratified_split_regression
    except ImportError:
        from ai_training.split_strategy import get_split_recommendation, stratified_split_regression
    
    # Get recommendation and do stratified split on indices
    recommendation = get_split_recommendation(len(all_targets), all_targets.max() - all_targets.min())
    
    # Create index array
    indices = np.arange(len(full_dataset))
    
    # Do stratified split on indices
    _, _, _, _, split_info = stratified_split_regression(
        indices.reshape(-1, 1),  # Dummy X (we only care about y)
        all_targets,
        train_split=train_split,
        n_bins=recommendation['n_bins'],
        random_seed=random_seed
    )
    
    # Get train and val indices from the split
    # Re-do the split to get actual indices (not dummy X)
    np.random.seed(random_seed)
    
    # Use same binning as split_info
    from .split_strategy import create_bins
    bin_assignments = create_bins(all_targets, n_bins=recommendation['n_bins'], method='quantile')
    unique_bins = np.unique(bin_assignments)
    
    train_indices = []
    val_indices = []
    
    for bin_idx in unique_bins:
        bin_mask = bin_assignments == bin_idx
        bin_idxs = np.where(bin_mask)[0]
        np.random.shuffle(bin_idxs)
        
        split_point = int(len(bin_idxs) * train_split)
        train_indices.extend(bin_idxs[:split_point].tolist())
        val_indices.extend(bin_idxs[split_point:].tolist())
    
    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
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
    
    # Get train and val target values AND image IDs for distribution info
    train_targets = [all_targets[i] for i in train_indices]
    val_targets = [all_targets[i] for i in val_indices]
    
    # Get actual image IDs for train and val sets
    # IMPORTANT: train_indices/val_indices are positions in full_dataset (filtered)
    # Must map through full_dataset.valid_indices to get positions in original images_data
    train_image_ids = []
    for i in train_indices:
        if i < len(full_dataset.valid_indices):
            original_idx = full_dataset.valid_indices[i]
            if original_idx < len(images_data) and 'id' in images_data[original_idx]:
                train_image_ids.append(images_data[original_idx]['id'])
    
    val_image_ids = []
    for i in val_indices:
        if i < len(full_dataset.valid_indices):
            original_idx = full_dataset.valid_indices[i]
            if original_idx < len(images_data) and 'id' in images_data[original_idx]:
                val_image_ids.append(images_data[original_idx]['id'])
    
    stats = {
        "total_samples": len(full_dataset),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "batch_size": batch_size,
        "split_strategy": recommendation['strategy'],
        "n_bins": recommendation['n_bins'],
        "split_info": split_info,
        "train_target_range": [float(min(train_targets)), float(max(train_targets))],
        "val_target_range": [float(min(val_targets)), float(max(val_targets))],
        "train_image_ids": train_image_ids,
        "val_image_ids": val_image_ids
    }
    
    return train_loader, val_loader, stats


class AugmentedDrawingDataset(Dataset):
    """
    PyTorch Dataset for augmented training data from disk.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Initialize dataset from augmented data directory.
        
        Args:
            data_dir: Directory containing augmented data
            split: 'train' or 'val'
            transform: Optional transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Load all image-label pairs
        self.samples = []
        label_files = sorted(self.split_dir.glob("*.json"))
        
        for label_file in label_files:
            # Check if corresponding image exists
            img_file = label_file.with_suffix('.png')
            if img_file.exists():
                self.samples.append({
                    'image_path': img_file,
                    'label_path': label_file
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        sample = self.samples[idx]
        
        # Load image
        img_array = cv2.imread(str(sample['image_path']), cv2.IMREAD_GRAYSCALE)
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Load label
        with open(sample['label_path'], 'r') as f:
            label_data = json.load(f)
        
        target_value = float(label_data['target_value'])
        target_tensor = torch.tensor([target_value], dtype=torch.float32)
        
        return img_tensor, target_tensor


def create_augmented_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    shuffle_train: bool = True,
    transform=None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create dataloaders from augmented data directory.
    
    Args:
        data_dir: Directory containing augmented train/val data
        batch_size: Batch size
        shuffle_train: Whether to shuffle training data
        transform: Optional transforms
    
    Returns:
        (train_loader, val_loader, stats)
    """
    # Create datasets
    train_dataset = AugmentedDrawingDataset(data_dir, split='train', transform=transform)
    val_dataset = AugmentedDrawingDataset(data_dir, split='val', transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load metadata
    metadata_file = Path(data_dir) / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    stats = {
        "total_samples": len(train_dataset) + len(val_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "batch_size": batch_size,
        "augmentation_config": metadata.get('augmentation_config', {}),
        "statistics": metadata.get('statistics', {})
    }
    
    return train_loader, val_loader, stats

