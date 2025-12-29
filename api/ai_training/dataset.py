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

from .normalization import TargetNormalizer


class DrawingDataset(Dataset):
    """
    PyTorch Dataset for training data images.
    """
    
    def __init__(
        self,
        images_data: List[Dict],
        target_feature: str,
        transform=None,
        normalizer: Optional[TargetNormalizer] = None,
        is_classification: bool = False,
        num_classes: int = None
    ):
        """
        Initialize dataset.
        
        Args:
            images_data: List of dicts with 'processed_image_data', 'features_data'
            target_feature: Name of feature to predict (e.g., 'Total_Score' or 'Custom_Class_5')
            transform: Optional torchvision transforms
            normalizer: Optional TargetNormalizer for target values (None for classification)
            is_classification: True if classification mode
            num_classes: Number of classes (for classification)
        """
        self.images_data = images_data
        self.target_feature = target_feature
        self.transform = transform
        self.normalizer = normalizer
        self.is_classification = is_classification
        self.num_classes = num_classes
        
        # Validate classification parameters
        if is_classification and num_classes is None:
            raise ValueError(f"num_classes must be provided when is_classification=True for feature '{target_feature}'")
        
        # Filter: Only keep images that have the target feature
        self.valid_indices = []
        for idx, img_data in enumerate(images_data):
            features = {}
            if img_data.get('features_data'):
                try:
                    features = json.loads(img_data['features_data'])
                except:
                    pass
            
            # Check if feature exists
            if is_classification:
                # For Custom_Class, check if classification exists
                if "Custom_Class" in features and str(num_classes) in features.get("Custom_Class", {}):
                    self.valid_indices.append(idx)
            else:
                # For regression, check if feature exists
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
        
        if self.is_classification:
            # Classification mode: Read Custom_Class label
            custom_class = features.get("Custom_Class", {})
            class_data = custom_class.get(str(self.num_classes))
            
            if class_data:
                target_value = int(class_data["label"])
            else:
                target_value = 0  # Fallback
            
            # NO normalization for classification!
            # CrossEntropyLoss expects class indices (Long tensor, scalar)
            target_tensor = torch.tensor(target_value, dtype=torch.long)
        else:
            # Regression mode: Read numeric feature
            target_value = float(features[self.target_feature])
            
            # Apply normalization if normalizer is provided
            if self.normalizer is not None:
                target_value = self.normalizer.transform(np.array([target_value]))[0]
            
            target_tensor = torch.tensor([target_value], dtype=torch.float32)
        
        return img_tensor, target_tensor


def create_dataloaders(
    images_data: List[Dict],
    target_feature: str,
    train_split: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    random_seed: int = 42,
    normalizer: Optional[TargetNormalizer] = None,
    is_classification: bool = False,
    num_classes: int = None
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
    # Create full dataset WITHOUT normalizer to get raw target values for stratified split
    full_dataset_raw = DrawingDataset(
        images_data, 
        target_feature, 
        normalizer=None,
        is_classification=is_classification,
        num_classes=num_classes
    )
    
    if len(full_dataset_raw) == 0:
        raise ValueError(f"No samples with feature '{target_feature}'")
    
    # Extract all target values (raw, unnormalized) for stratified split
    all_targets = []
    for i in range(len(full_dataset_raw)):
        _, target = full_dataset_raw[i]
        all_targets.append(target.item())
    
    all_targets = np.array(all_targets)
    
    # Fit normalizer if provided
    if normalizer is not None:
        normalizer.fit(all_targets)
        print(f"   Fitted normalizer on {len(all_targets)} samples")
    
    # Import split strategy
    try:
        from .split_strategy import get_split_recommendation, stratified_split_regression, stratified_split_classification
    except ImportError:
        from ai_training.split_strategy import get_split_recommendation, stratified_split_regression, stratified_split_classification
    
    # Create index array
    indices = np.arange(len(full_dataset_raw))
    
    # Initialize variables that may be used later
    recommendation = None
    split_info = None
    
    # Choose split strategy based on task type
    if is_classification:
        # For classification: stratify by actual class labels
        print(f"   Using CLASSIFICATION stratification (by class labels)")
        stratified_split_succeeded = False
        try:
            _, _, _, _, split_info = stratified_split_classification(
                indices.reshape(-1, 1),  # Dummy X
                all_targets,
                train_split=train_split,
                random_seed=random_seed
            )
            stratified_split_succeeded = True
        except Exception as e:
            print(f"   Warning: Stratified classification split failed: {e}, falling back to random split")
            split_info = {
                'method': 'random',
                'reason': 'stratified_classification_split_failed'
            }
        
        if stratified_split_succeeded:
            # Re-do stratified split to get actual indices
            np.random.seed(random_seed)
            unique_classes = np.unique(all_targets)
            
            train_indices = []
            val_indices = []
            
            for cls in unique_classes:
                class_mask = all_targets == cls
                class_idxs = np.where(class_mask)[0]
                np.random.shuffle(class_idxs)
                
                split_point = int(len(class_idxs) * train_split)
                
                # Ensure at least 1 sample in test if possible
                if split_point == len(class_idxs) and len(class_idxs) > 1:
                    split_point = len(class_idxs) - 1
                
                train_indices.extend(class_idxs[:split_point].tolist())
                val_indices.extend(class_idxs[split_point:].tolist())
            
            # Create recommendation dict for classification
            recommendation = {
                'strategy': 'stratified_classification',
                'n_bins': len(unique_classes)
            }
        else:
            # Fall back to random split (consistent with regression path)
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            split_point = int(len(indices) * train_split)
            train_indices = indices[:split_point].tolist()
            val_indices = indices[split_point:].tolist()
            
            # Create recommendation dict for random split
            recommendation = {
                'strategy': 'random',
                'n_bins': len(np.unique(all_targets))
            }
        
    else:
        # For regression: stratify by binning continuous values
        print(f"   Using REGRESSION stratification (by value bins)")
        
        # Validate all_targets
        if len(all_targets) == 0:
            raise ValueError(f"No valid target values found for feature '{target_feature}'")
        
        # Calculate value range
        target_min = all_targets.min()
        target_max = all_targets.max()
        value_range = target_max - target_min
        
        # If all values are the same, use simple random split
        if value_range == 0:
            print(f"   Warning: All target values are identical ({target_min}), using random split")
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            split_point = int(len(indices) * train_split)
            train_indices = indices[:split_point].tolist()
            val_indices = indices[split_point:].tolist()
            # Create a simple split_info for stats
            split_info = {
                'method': 'random',
                'reason': 'all_values_identical'
            }
            recommendation = {
                'strategy': 'random',
                'n_bins': 1
            }
        else:
            # Get split recommendation
            try:
                recommendation = get_split_recommendation(len(all_targets), value_range)
                n_bins = recommendation['n_bins']
            except Exception as e:
                print(f"   Warning: Failed to get split recommendation: {e}, using default n_bins=5")
                n_bins = 5
            
            try:
                _, _, _, _, split_info = stratified_split_regression(
                    indices.reshape(-1, 1),  # Dummy X
                    all_targets,
                    train_split=train_split,
                    n_bins=n_bins,
                    random_seed=random_seed
                )
            except Exception as e:
                print(f"   Warning: Stratified split failed: {e}, falling back to random split")
                np.random.seed(random_seed)
                np.random.shuffle(indices)
                split_point = int(len(indices) * train_split)
                train_indices = indices[:split_point].tolist()
                val_indices = indices[split_point:].tolist()
                # Create a simple split_info for stats
                split_info = {
                    'method': 'random',
                    'reason': 'stratified_split_failed'
                }
                if recommendation is None:
                    recommendation = {
                        'strategy': 'random',
                        'n_bins': n_bins
                    }
            else:
                # Re-do split to get actual indices
                np.random.seed(random_seed)
                
                from .split_strategy import create_bins
                bin_assignments = create_bins(all_targets, n_bins=n_bins, method='quantile')
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
    
    # Validate that split resulted in non-empty sets
    if len(train_indices) == 0:
        raise ValueError(f"No training samples found for feature '{target_feature}'. "
                         f"Total samples: {len(full_dataset_raw)}, "
                         f"Valid samples: {len(all_targets)}")
    if len(val_indices) == 0:
        raise ValueError(f"No validation samples found for feature '{target_feature}'. "
                         f"Total samples: {len(full_dataset_raw)}, "
                         f"Valid samples: {len(all_targets)}")
    
    # Create new datasets WITH normalizer for training
    full_dataset_normalized = DrawingDataset(
        images_data, 
        target_feature, 
        normalizer=normalizer,
        is_classification=is_classification,
        num_classes=num_classes
    )
    
    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(full_dataset_normalized, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_normalized, val_indices)
    
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
    # IMPORTANT: train_indices/val_indices are positions in full_dataset_normalized (filtered)
    # Must map through full_dataset_normalized.valid_indices to get positions in original images_data
    train_image_ids = []
    for i in train_indices:
        if i < len(full_dataset_normalized.valid_indices):
            original_idx = full_dataset_normalized.valid_indices[i]
            if original_idx < len(images_data) and 'id' in images_data[original_idx]:
                train_image_ids.append(images_data[original_idx]['id'])
    
    val_image_ids = []
    for i in val_indices:
        if i < len(full_dataset_normalized.valid_indices):
            original_idx = full_dataset_normalized.valid_indices[i]
            if original_idx < len(images_data) and 'id' in images_data[original_idx]:
                val_image_ids.append(images_data[original_idx]['id'])
    
    stats = {
        "total_samples": len(full_dataset_normalized),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "batch_size": batch_size,
        "train_target_range": [float(min(train_targets)), float(max(train_targets))],
        "val_target_range": [float(min(val_targets)), float(max(val_targets))],
        "train_image_ids": train_image_ids,
        "val_image_ids": val_image_ids
    }
    
    # Add split strategy info if available
    if recommendation is not None:
        stats["split_strategy"] = recommendation.get('strategy', 'unknown')
        stats["n_bins"] = recommendation.get('n_bins', None)
    else:
        stats["split_strategy"] = 'unknown'
        stats["n_bins"] = None
    
    if split_info is not None:
        stats["split_info"] = split_info
    else:
        stats["split_info"] = {'method': 'unknown'}
    
    return train_loader, val_loader, stats


def create_dataloaders_from_ids(
    images_data: List[Dict],
    target_feature: str,
    train_image_ids: List[int],
    val_image_ids: List[int],
    batch_size: int = 8,
    shuffle: bool = True,
    normalizer: Optional[TargetNormalizer] = None,
    is_classification: bool = False,
    num_classes: int = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders from specific image IDs.
    This is used for testing models on the same train/val split as during training.
    
    Args:
        images_data: List of image data dicts (must include 'id' field)
        target_feature: Feature name to predict
        train_image_ids: List of image IDs for training set
        val_image_ids: List of image IDs for validation set
        batch_size: Batch size
        shuffle: Shuffle training data
        normalizer: Optional TargetNormalizer for target values
        is_classification: True if classification mode
        num_classes: Number of classes (for classification)
    
    Returns:
        (train_loader, val_loader, stats)
    """
    # Create a mapping from image ID to index in images_data
    id_to_idx = {img['id']: idx for idx, img in enumerate(images_data) if 'id' in img}
    
    # Filter images_data to only include train and val IDs
    train_indices_in_data = []
    val_indices_in_data = []
    
    for img_id in train_image_ids:
        if img_id in id_to_idx:
            train_indices_in_data.append(id_to_idx[img_id])
    
    for img_id in val_image_ids:
        if img_id in id_to_idx:
            val_indices_in_data.append(id_to_idx[img_id])
    
    if len(train_indices_in_data) == 0:
        raise ValueError(f"No training images found for the provided train_image_ids. "
                         f"Requested {len(train_image_ids)} IDs, found {len(train_indices_in_data)}")
    
    if len(val_indices_in_data) == 0:
        raise ValueError(f"No validation images found for the provided val_image_ids. "
                         f"Requested {len(val_image_ids)} IDs, found {len(val_indices_in_data)}")
    
    # Create datasets with only the specified images
    train_images_data = [images_data[i] for i in train_indices_in_data]
    val_images_data = [images_data[i] for i in val_indices_in_data]
    
    # Create datasets
    train_dataset = DrawingDataset(
        train_images_data,
        target_feature,
        normalizer=normalizer,
        is_classification=is_classification,
        num_classes=num_classes
    )
    
    val_dataset = DrawingDataset(
        val_images_data,
        target_feature,
        normalizer=normalizer,
        is_classification=is_classification,
        num_classes=num_classes
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    
    # Extract target values for stats
    train_targets = []
    val_targets = []
    
    for img_data in train_images_data:
        try:
            features = json.loads(img_data.get('features_data', '{}'))
            if is_classification:
                custom_class = features.get("Custom_Class", {})
                class_data = custom_class.get(str(num_classes))
                if class_data:
                    train_targets.append(int(class_data["label"]))
            else:
                if target_feature in features:
                    train_targets.append(float(features[target_feature]))
        except:
            pass
    
    for img_data in val_images_data:
        try:
            features = json.loads(img_data.get('features_data', '{}'))
            if is_classification:
                custom_class = features.get("Custom_Class", {})
                class_data = custom_class.get(str(num_classes))
                if class_data:
                    val_targets.append(int(class_data["label"]))
            else:
                if target_feature in features:
                    val_targets.append(float(features[target_feature]))
        except:
            pass
    
    stats = {
        "total_samples": len(train_dataset) + len(val_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "batch_size": batch_size,
        "split_strategy": "from_metadata",
        "n_bins": len(np.unique(train_targets + val_targets)) if train_targets and val_targets else None,
        "split_info": {
            'method': 'from_metadata',
            'train_image_ids': train_image_ids,
            'val_image_ids': val_image_ids
        },
        "train_target_range": [float(min(train_targets)), float(max(train_targets))] if train_targets else [],
        "val_target_range": [float(min(val_targets)), float(max(val_targets))] if val_targets else [],
        "train_image_ids": train_image_ids,
        "val_image_ids": val_image_ids
    }
    
    return train_loader, val_loader, stats


class AugmentedDrawingDataset(Dataset):
    """
    PyTorch Dataset for augmented training data from disk.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, is_classification: bool = False):
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
        self.is_classification = is_classification
        
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
        
        # Use explicit is_classification flag instead of heuristic
        if self.is_classification:
            # Classification: scalar long tensor
            target_tensor = torch.tensor(int(target_value), dtype=torch.long)
        else:
            # Regression: [1] float tensor
            target_tensor = torch.tensor([target_value], dtype=torch.float32)
        
        return img_tensor, target_tensor


def create_augmented_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    shuffle_train: bool = True,
    transform=None,
    is_classification: bool = False
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
    train_dataset = AugmentedDrawingDataset(data_dir, split='train', transform=transform, is_classification=is_classification)
    val_dataset = AugmentedDrawingDataset(data_dir, split='val', transform=transform, is_classification=is_classification)
    
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
        "statistics": metadata.get('statistics', {}),
        # Restore split strategy information
        "split_strategy": metadata.get('split_strategy', 'unknown'),
        "split_info": metadata.get('split_info', {}),
        "n_bins": metadata.get('n_bins', 0),
        # Restore train/val image IDs for testing on same split
        "train_image_ids": metadata.get('train_image_ids', []),
        "val_image_ids": metadata.get('val_image_ids', [])
    }
    
    return train_loader, val_loader, stats

