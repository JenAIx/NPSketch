"""
Dataset and DataLoader for Training Data Images

Supports loading from:
1. Database (original implementation)
2. Augmented data directory (new)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import io
import cv2
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

from .normalization import TargetNormalizer
from .preprocessing import preprocess_for_training, resize_for_model_input, get_model_input_size
from utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel target_feature that selects the 60-sub-label component model.
COMPONENTS_TARGET = "Components"


def components_to_vector(features: dict):
    """
    Flatten features_data["components"] to a 60-vector in the canonical column
    order used everywhere: per element e (0..19) -> presence, accuracy, position.
    Returns None if the row has no component sub-labels.
    """
    c = features.get("components")
    if not c:
        return None
    pres, acc, pos = c["presence"], c["accuracy"], c["position"]
    vec = []
    for e in range(20):
        vec.extend([float(pres[e]), float(acc[e]), float(pos[e])])
    return vec


class DrawingDataset(Dataset):
    """
    PyTorch Dataset for training data images.
    
    Supports optional pre-shrink for consistent preprocessing with augmented data.
    """
    
    def __init__(
        self,
        images_data: List[Dict],
        target_feature: str,
        transform=None,
        normalizer: Optional[TargetNormalizer] = None,
        is_classification: bool = False,
        num_classes: int = None,
        pre_shrink_enabled: bool = True,
        pre_shrink_factor: float = 0.90,
        is_components: bool = False
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
            pre_shrink_enabled: Apply pre-shrink preprocessing (default: True)
            pre_shrink_factor: Pre-shrink factor (default: 0.90 = 10% shrink)
        """
        self.images_data = images_data
        self.target_feature = target_feature
        self.transform = transform
        self.normalizer = normalizer
        self.is_classification = is_classification
        self.is_components = is_components
        self.num_classes = num_classes
        self.pre_shrink_enabled = pre_shrink_enabled
        self.pre_shrink_factor = pre_shrink_factor
        # CNN input resolution (downscale from 568x274); None = full res
        self.model_input_size = get_model_input_size()

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
            if is_components:
                # Component mode: needs the 60 sub-labels
                if features.get("components"):
                    self.valid_indices.append(idx)
            elif is_classification:
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
        """
        Get item by index.
        
        Preprocessing pipeline (matches augmented path via preprocess_for_training):
        1. Load image from bytes, convert to RGB
        2. Pre-shrink (if enabled) - creates margins for augmentation tolerance
        3. Binarization - removes anti-aliasing artifacts from shrink
        4. Line normalization - ensures consistent 2px line thickness
        5. Convert to grayscale for model input
        6. Normalize to [0, 1] float32
        """
        real_idx = self.valid_indices[idx]
        img_data = self.images_data[real_idx]
        
        # Load image from bytes
        image_bytes = img_data['processed_image_data']
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB for preprocessing pipeline
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array (uint8)
        img_array = np.array(image, dtype=np.uint8)
        
        # Apply full preprocessing pipeline (uses shared functions from preprocessing.py)
        # This ensures consistency with augmented training path
        img_array = preprocess_for_training(
            img_array,
            pre_shrink_enabled=self.pre_shrink_enabled,
            pre_shrink_factor=self.pre_shrink_factor,
            apply_binarize=True,
            apply_line_norm=True,
            convert_to_grayscale=True
        )
        
        # Convert to float32 and normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Downscale to CNN input resolution (identical in inference)
        img_array = resize_for_model_input(img_array, self.model_input_size)

        # Add channel dimension: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Get target value
        features = json.loads(img_data['features_data'])

        if self.is_components:
            # Multi-label: 60-dim float vector (BCE target), no normalization
            vec = components_to_vector(features)
            target_tensor = torch.tensor(vec, dtype=torch.float32)
        elif self.is_classification:
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


def build_regression_imbalance_sampler(
    train_targets: np.ndarray
) -> Tuple[Optional[WeightedRandomSampler], Dict]:
    """
    Build a WeightedRandomSampler that oversamples rare target-value bins
    (regression only). Counteracts heavily skewed score distributions where
    plain MSE training collapses to predicting the dominant range.

    Targets may be on any scale (raw or normalized) - bins are equal-width
    over the observed range. Configured via training_config.yaml
    (training.regression.imbalance). Returns (None, info) when disabled or
    not applicable.

    Returns:
        (sampler_or_none, info_dict_for_metadata)
    """
    from config import get_config
    cfg = get_config().get('training.regression.imbalance', {}) or {}

    info = {'enabled': False}
    if not cfg.get('enabled', False):
        return None, info

    train_targets = np.asarray(train_targets, dtype=np.float64)
    num_bins = int(cfg.get('num_bins', 6))
    max_weight = float(cfg.get('max_weight', 10.0))

    value_range = train_targets.max() - train_targets.min()
    if len(train_targets) < num_bins * 2 or value_range == 0:
        logger.info("Imbalance sampler: not applicable (too few samples or constant targets)")
        return None, info

    # Equal-width bins over the observed range
    edges = np.linspace(train_targets.min(), train_targets.max(), num_bins + 1)
    bin_idx = np.clip(np.digitize(train_targets, edges[1:-1]), 0, num_bins - 1)
    counts = np.bincount(bin_idx, minlength=num_bins)
    nonempty = int((counts > 0).sum())

    # Inverse-frequency weight per bin, capped
    bin_weights = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        if counts[b] > 0:
            bin_weights[b] = min(len(train_targets) / (nonempty * counts[b]), max_weight)
    sample_weights = bin_weights[bin_idx]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_targets),
        replacement=True
    )

    info = {
        'enabled': True,
        'num_bins': num_bins,
        'max_weight': max_weight,
        'bin_edges': [float(e) for e in edges],
        'bin_counts': [int(c) for c in counts],
        'bin_weights': [round(float(w), 4) for w in bin_weights]
    }
    logger.info("Imbalance sampler enabled (WeightedRandomSampler, regression):")
    for b in range(num_bins):
        logger.info(f"  Bin [{edges[b]:.2f}, {edges[b+1]:.2f}]: "
                    f"{counts[b]} samples, weight {bin_weights[b]:.4f}")
    return sampler, info


def create_dataloaders(
    images_data: List[Dict],
    target_feature: str,
    train_split: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    random_seed: int = 42,
    normalizer: Optional[TargetNormalizer] = None,
    is_classification: bool = False,
    num_classes: int = None,
    pre_shrink_enabled: bool = True,
    pre_shrink_factor: float = 0.90
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
        normalizer: Optional target normalizer
        is_classification: Whether this is classification task
        num_classes: Number of classes (for classification)
        pre_shrink_enabled: Apply pre-shrink preprocessing (default: True)
        pre_shrink_factor: Pre-shrink factor (default: 0.90 = 10% shrink)
    
    Returns:
        (train_loader, val_loader, stats)
    """
    # Log pre-shrink config
    logger.info(f"Non-augmented dataloader: pre_shrink_enabled={pre_shrink_enabled}, factor={pre_shrink_factor}")
    
    # Create full dataset WITHOUT normalizer to get raw target values for stratified split
    full_dataset_raw = DrawingDataset(
        images_data, 
        target_feature, 
        normalizer=None,
        is_classification=is_classification,
        num_classes=num_classes,
        pre_shrink_enabled=pre_shrink_enabled,
        pre_shrink_factor=pre_shrink_factor
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
        logger.info(f"Fitted normalizer on {len(all_targets)} samples")
    
    # Import split strategy
    try:
        from .split_strategy import get_split_recommendation, stratified_group_split
    except ImportError:
        from ai_training.split_strategy import get_split_recommendation, stratified_group_split

    # Build patient-level group keys, aligned with the filtered dataset positions.
    # All images of the same patient_id are assigned to the SAME set (no leakage).
    # Images without patient_id form their own single-image group.
    groups = []
    for i in range(len(full_dataset_raw)):
        orig_idx = full_dataset_raw.valid_indices[i]
        pid = images_data[orig_idx].get('patient_id')
        groups.append(str(pid) if pid else f"__img_{images_data[orig_idx].get('id', orig_idx)}")

    # Initialize variables that may be used later
    recommendation = None
    split_info = None

    # Choose split strategy based on task type
    if is_classification:
        # For classification: stratify by majority class per patient group
        logger.info("Using CLASSIFICATION stratification (patient-level groups)")
        try:
            train_indices, val_indices, split_info = stratified_group_split(
                all_targets,
                groups,
                train_split=train_split,
                random_seed=random_seed,
                is_classification=True
            )
            recommendation = {
                'strategy': 'stratified_group_classification',
                'n_bins': len(np.unique(all_targets))
            }
        except Exception as e:
            logger.warning(f"Grouped classification split failed: {e}, "
                           f"falling back to grouped random split")
            train_indices, val_indices, split_info = stratified_group_split(
                all_targets,
                groups,
                train_split=train_split,
                n_bins=1,
                random_seed=random_seed,
                is_classification=False
            )
            split_info['method'] = 'grouped_random'
            split_info['reason'] = 'stratified_group_classification_failed'
            recommendation = {
                'strategy': 'grouped_random',
                'n_bins': len(np.unique(all_targets))
            }

    else:
        # For regression: stratify by quantile bins of patient-median values
        logger.info("Using REGRESSION stratification (patient-level groups)")

        # Validate all_targets
        if len(all_targets) == 0:
            raise ValueError(f"No valid target values found for feature '{target_feature}'")

        value_range = all_targets.max() - all_targets.min()

        try:
            recommendation = get_split_recommendation(len(all_targets), value_range)
            n_bins = recommendation['n_bins']
        except Exception as e:
            logger.warning(f"Failed to get split recommendation: {e}, using default n_bins=5")
            n_bins = 5
        recommendation = {'strategy': 'stratified_group', 'n_bins': n_bins}

        try:
            train_indices, val_indices, split_info = stratified_group_split(
                all_targets,
                groups,
                train_split=train_split,
                n_bins=n_bins,
                random_seed=random_seed,
                is_classification=False
            )
        except Exception as e:
            logger.warning(f"Grouped stratified split failed: {e}, "
                           f"falling back to grouped random split")
            train_indices, val_indices, split_info = stratified_group_split(
                all_targets,
                groups,
                train_split=train_split,
                n_bins=1,
                random_seed=random_seed,
                is_classification=False
            )
            split_info['method'] = 'grouped_random'
            split_info['reason'] = 'stratified_group_split_failed'
            recommendation['strategy'] = 'grouped_random'
    
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
        num_classes=num_classes,
        pre_shrink_enabled=pre_shrink_enabled,
        pre_shrink_factor=pre_shrink_factor
    )
    
    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(full_dataset_normalized, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset_normalized, val_indices)

    # Optional oversampling of rare score bins (regression only)
    imbalance_sampler = None
    imbalance_info = {'enabled': False}
    if not is_classification:
        imbalance_sampler, imbalance_info = build_regression_imbalance_sampler(
            np.array([all_targets[i] for i in train_indices])
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if imbalance_sampler is None else False,
        sampler=imbalance_sampler,
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
        "val_image_ids": val_image_ids,
        # Pre-shrink config (for model metadata, enables prediction to apply same)
        "pre_shrink": {
            "enabled": pre_shrink_enabled,
            "factor": pre_shrink_factor
        },
        "imbalance_sampler": imbalance_info
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
    
    # Calculate class weights for classification
    if is_classification and num_classes is not None and split_info is not None:
        train_dist = split_info.get('train_distribution', {})
        train_counts = train_dist.get('class_counts', {})
        
        if train_counts:
            total_samples = len(train_indices)
            class_weights = []
            
            for cls_id in range(num_classes):
                count = train_counts.get(cls_id, 0)
                if count > 0:
                    weight = total_samples / (num_classes * count)
                else:
                    weight = 1.0
                class_weights.append(weight)
            
            stats['class_weights'] = class_weights
            logger.info(f"\n>> Class weights calculated: {[f'{w:.4f}' for w in class_weights]}")
        else:
            stats['class_weights'] = None
    else:
        stats['class_weights'] = None
    
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
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, is_classification: bool = False,
                 is_components: bool = False):
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
        self.is_components = is_components
        # CNN input resolution (downscale from 568x274); None = full res
        self.model_input_size = get_model_input_size()
        
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

        # Downscale to CNN input resolution (identical in inference)
        img_array = resize_for_model_input(img_array, self.model_input_size)

        # Convert to tensor: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Load label
        with open(sample['label_path'], 'r') as f:
            label_data = json.load(f)

        # Use explicit mode flags instead of heuristics
        if self.is_components:
            # Multi-label: 60-dim float vector persisted by the augmenter
            target_tensor = torch.tensor(label_data['target_vector'], dtype=torch.float32)
        elif self.is_classification:
            # Classification: scalar long tensor
            target_tensor = torch.tensor(int(float(label_data['target_value'])), dtype=torch.long)
        else:
            # Regression: [1] float tensor
            target_tensor = torch.tensor([float(label_data['target_value'])], dtype=torch.float32)

        return img_tensor, target_tensor


def create_augmented_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    shuffle_train: bool = True,
    transform=None,
    is_classification: bool = False,
    is_components: bool = False
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
    train_dataset = AugmentedDrawingDataset(data_dir, split='train', transform=transform,
                                            is_classification=is_classification, is_components=is_components)
    val_dataset = AugmentedDrawingDataset(data_dir, split='val', transform=transform,
                                          is_classification=is_classification, is_components=is_components)

    # Optional oversampling of rare score bins (regression only — not for
    # classification or the multi-label component head).
    imbalance_sampler = None
    imbalance_info = {'enabled': False}
    if not is_classification and not is_components:
        train_targets = []
        for sample in train_dataset.samples:
            try:
                with open(sample['label_path'], 'r') as f:
                    train_targets.append(float(json.load(f)['target_value']))
            except Exception as e:
                logger.warning(f"Could not read label {sample['label_path']}: {e}")
                train_targets.append(0.0)
        imbalance_sampler, imbalance_info = build_regression_imbalance_sampler(
            np.array(train_targets)
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train if imbalance_sampler is None else False,
        sampler=imbalance_sampler,
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
        "val_image_ids": metadata.get('val_image_ids', []),
        "imbalance_sampler": imbalance_info
    }
    
    # Calculate class weights for classification from metadata
    if is_classification:
        split_info = metadata.get('split_info', {})
        train_dist = split_info.get('train_distribution', {})
        train_counts = train_dist.get('class_counts', {})
        num_classes = split_info.get('num_classes', 0)
        
        if train_counts and num_classes > 0:
            # Use original (non-augmented) sample count from metadata for consistency
            # train_counts contains original class counts, so we need original total_samples
            original_total_samples = train_dist.get('count', 0)
            
            if original_total_samples == 0:
                # Fallback: try to calculate from class counts
                original_total_samples = sum(train_counts.values())
            
            class_weights = []
            
            logger.info("="*60)
            logger.info("CLASS DISTRIBUTION ANALYSIS (Augmented Data)")
            logger.info("="*60)
            logger.info(f"Original train samples (non-augmented): {original_total_samples}")
            logger.info(f"Augmented train samples: {len(train_dataset)}")
            
            for cls_id in range(num_classes):
                # Try both string and int keys
                count = train_counts.get(str(cls_id), train_counts.get(cls_id, 0))
                if count > 0:
                    # Use original total_samples and original class counts for consistency
                    weight = original_total_samples / (num_classes * count)
                    pct = (count / original_total_samples) * 100 if original_total_samples > 0 else 0
                    logger.info(f"  Class {cls_id}: {count} original samples ({pct:.1f}% of original)")
                else:
                    weight = 1.0
                    logger.info(f"  Class {cls_id}: 0 samples (WARNING!)")
                class_weights.append(weight)
            
            logger.info("\nCalculated class weights (inverse frequency, based on original distribution):")
            for cls_id, weight in enumerate(class_weights):
                logger.info(f"  Class {cls_id}: weight = {weight:.4f}")
            logger.info("="*60)
            
            stats['class_weights'] = class_weights
        else:
            logger.warning("Could not calculate class weights from metadata")
            stats['class_weights'] = None
    else:
        stats['class_weights'] = None
    
    return train_loader, val_loader, stats

