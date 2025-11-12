"""
Data Augmentation Library for Training Data

Creates augmented versions of training images to increase dataset size
and improve model generalization.

Augmentations:
- Rotation: ±1-3 degrees (simulates paper tilt)
- Translation: ±5-10 pixels (simulates position shifts)
- Scaling: 95-105% (simulates size variations)

All augmentations preserve:
- Black background
- Line quality (no interpolation artifacts)
- Aspect ratio
- Image dimensions
"""

import numpy as np
import cv2
from PIL import Image
import io
from typing import List, Dict, Tuple
import os
import json
import shutil
from pathlib import Path


class ImageAugmentor:
    """
    Augments training images with realistic variations.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-3.0, 3.0),
        translation_range: Tuple[int, int] = (-10, 10),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        num_augmentations: int = 5
    ):
        """
        Initialize augmentor.
        
        Args:
            rotation_range: Min/max rotation in degrees
            translation_range: Min/max translation in pixels
            scale_range: Min/max scale factor
            num_augmentations: Number of augmented versions per image
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.num_augmentations = num_augmentations
    
    def augment_image(
        self,
        image: np.ndarray,
        rotation: float = None,
        tx: int = None,
        ty: int = None,
        scale: float = None,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image (numpy array, grayscale or RGB)
            rotation: Rotation angle in degrees (random if None)
            tx: Translation X in pixels (random if None)
            ty: Translation Y in pixels (random if None)
            scale: Scale factor (random if None)
            random_seed: Random seed for reproducibility
        
        Returns:
            Augmented image
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random parameters if not provided
        if rotation is None:
            rotation = np.random.uniform(*self.rotation_range)
        if tx is None:
            tx = np.random.randint(*self.translation_range)
        if ty is None:
            ty = np.random.randint(*self.translation_range)
        if scale is None:
            scale = np.random.uniform(*self.scale_range)
        
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Create combined transformation matrix
        # cv2.getRotationMatrix2D returns 2x3 matrix for rotation and scale
        M = cv2.getRotationMatrix2D(center, rotation, scale)
        
        # Add translation to the transformation matrix
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        # Use INTER_LINEAR for smooth edges and black background
        augmented = cv2.warpAffine(
            image,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0  # Black background
        )
        
        return augmented
    
    def augment_batch(
        self,
        image: np.ndarray,
        num_augmentations: int = None
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Create multiple augmented versions of an image.
        
        Args:
            image: Input image
            num_augmentations: Number of augmentations (uses self.num_augmentations if None)
        
        Returns:
            List of (augmented_image, parameters) tuples
        """
        if num_augmentations is None:
            num_augmentations = self.num_augmentations
        
        augmented_images = []
        
        for i in range(num_augmentations):
            # Generate random parameters
            rotation = np.random.uniform(*self.rotation_range)
            tx = np.random.randint(*self.translation_range)
            ty = np.random.randint(*self.translation_range)
            scale = np.random.uniform(*self.scale_range)
            
            # Apply augmentation
            aug_img = self.augment_image(image, rotation, tx, ty, scale)
            
            # Store parameters
            params = {
                'rotation': float(rotation),
                'translation_x': int(tx),
                'translation_y': int(ty),
                'scale': float(scale)
            }
            
            augmented_images.append((aug_img, params))
        
        return augmented_images


class AugmentedDatasetBuilder:
    """
    Builds augmented training/test datasets and saves to disk.
    """
    
    def __init__(
        self,
        output_dir: str,
        augmentor: ImageAugmentor = None,
        include_original: bool = True
    ):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save augmented data
            augmentor: ImageAugmentor instance (creates default if None)
            include_original: Whether to include original images
        """
        self.output_dir = Path(output_dir)
        self.augmentor = augmentor or ImageAugmentor()
        self.include_original = include_original
    
    def prepare_augmented_dataset(
        self,
        images_data: List[Dict],
        split_indices: Dict[str, List[int]],
        target_feature: str,
        clean_existing: bool = True
    ) -> Dict:
        """
        Prepare augmented dataset from training data.
        
        Args:
            images_data: List of image data dicts from database
            split_indices: Dict with 'train' and 'val' index lists
            target_feature: Feature name being predicted
            clean_existing: Whether to clean existing data
        
        Returns:
            Statistics about created dataset
        """
        # Create output directories
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        
        if clean_existing and self.output_dir.exists():
            print(f"Cleaning existing augmented data in {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'train': {'original': 0, 'augmented': 0, 'total': 0},
            'val': {'original': 0, 'augmented': 0, 'total': 0},
            'errors': []
        }
        
        # Process training set
        print(f"\n📊 Augmenting training set ({len(split_indices['train'])} images)...")
        self._process_split(
            images_data,
            split_indices['train'],
            train_dir,
            target_feature,
            stats['train'],
            stats['errors'],
            split_name='train'
        )
        
        # Process validation set
        print(f"\n📊 Augmenting validation set ({len(split_indices['val'])} images)...")
        self._process_split(
            images_data,
            split_indices['val'],
            val_dir,
            target_feature,
            stats['val'],
            stats['errors'],
            split_name='val'
        )
        
        # Save metadata
        metadata = {
            'target_feature': target_feature,
            'augmentation_config': {
                'rotation_range': self.augmentor.rotation_range,
                'translation_range': self.augmentor.translation_range,
                'scale_range': self.augmentor.scale_range,
                'num_augmentations': self.augmentor.num_augmentations
            },
            'include_original': self.include_original,
            'statistics': stats
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Augmented dataset created in {self.output_dir}")
        print(f"   Train: {stats['train']['total']} images ({stats['train']['original']} original + {stats['train']['augmented']} augmented)")
        print(f"   Val:   {stats['val']['total']} images ({stats['val']['original']} original + {stats['val']['augmented']} augmented)")
        
        if stats['errors']:
            print(f"\n⚠️ {len(stats['errors'])} errors occurred during augmentation")
        
        return stats
    
    def _process_split(
        self,
        images_data: List[Dict],
        indices: List[int],
        output_dir: Path,
        target_feature: str,
        stats: Dict,
        errors: List,
        split_name: str
    ):
        """Process a single split (train or val)."""
        for idx in indices:
            if idx >= len(images_data):
                errors.append(f"Index {idx} out of range for images_data")
                continue
            
            img_data = images_data[idx]
            
            try:
                # Get target value
                features = json.loads(img_data.get('features_data', '{}'))
                if target_feature not in features:
                    errors.append(f"Image {img_data.get('id', idx)} missing feature {target_feature}")
                    continue
                
                target_value = features[target_feature]
                
                # Load image
                image_bytes = img_data.get('processed_image_data')
                if not image_bytes:
                    errors.append(f"Image {img_data.get('id', idx)} has no image data")
                    continue
                
                pil_img = Image.open(io.BytesIO(image_bytes))
                img_array = np.array(pil_img)
                
                # Convert to grayscale if needed
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Get image ID
                img_id = img_data.get('id', idx)
                patient_id = img_data.get('patient_id', f'unknown_{idx}')
                
                # Save original image
                if self.include_original:
                    original_path = output_dir / f"{patient_id}_id{img_id}_original.png"
                    cv2.imwrite(str(original_path), img_array)
                    
                    # Save label
                    label_path = output_dir / f"{patient_id}_id{img_id}_original.json"
                    with open(label_path, 'w') as f:
                        json.dump({
                            'image_id': img_id,
                            'patient_id': patient_id,
                            'target_feature': target_feature,
                            'target_value': target_value,
                            'augmentation': None
                        }, f)
                    
                    stats['original'] += 1
                    stats['total'] += 1
                
                # Generate augmented versions
                augmented_images = self.augmentor.augment_batch(img_array)
                
                for aug_idx, (aug_img, aug_params) in enumerate(augmented_images):
                    # Save augmented image
                    aug_path = output_dir / f"{patient_id}_id{img_id}_aug{aug_idx}.png"
                    cv2.imwrite(str(aug_path), aug_img)
                    
                    # Save label with augmentation parameters
                    label_path = output_dir / f"{patient_id}_id{img_id}_aug{aug_idx}.json"
                    with open(label_path, 'w') as f:
                        json.dump({
                            'image_id': img_id,
                            'patient_id': patient_id,
                            'target_feature': target_feature,
                            'target_value': target_value,
                            'augmentation': aug_params
                        }, f)
                    
                    stats['augmented'] += 1
                    stats['total'] += 1
                
                # Progress indicator
                if stats['total'] % 10 == 0:
                    print(f"  Processed {stats['total']} images...")
                
            except Exception as e:
                errors.append(f"Error processing image {img_data.get('id', idx)}: {str(e)}")


def load_augmented_dataset(
    data_dir: str,
    split: str = 'train'
) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Load augmented dataset from disk.
    
    Args:
        data_dir: Directory containing augmented data
        split: 'train' or 'val'
    
    Returns:
        (images, targets, metadata_list)
    """
    split_dir = Path(data_dir) / split
    
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Load all images and labels
    images = []
    targets = []
    metadata_list = []
    
    # Get all JSON files (labels)
    label_files = sorted(split_dir.glob("*.json"))
    
    for label_file in label_files:
        # Load label
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        # Load corresponding image
        img_file = label_file.with_suffix('.png')
        if not img_file.exists():
            print(f"Warning: Image not found for {label_file.name}")
            continue
        
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        
        images.append(img)
        targets.append(label_data['target_value'])
        metadata_list.append(label_data)
    
    return images, targets, metadata_list


def get_augmentation_stats(data_dir: str) -> Dict:
    """
    Get statistics about augmented dataset.
    
    Args:
        data_dir: Directory containing augmented data
    
    Returns:
        Statistics dictionary
    """
    data_path = Path(data_dir)
    metadata_file = data_path / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    # Compute stats if metadata doesn't exist
    stats = {
        'train': {},
        'val': {}
    }
    
    for split in ['train', 'val']:
        split_dir = data_path / split
        if split_dir.exists():
            json_files = list(split_dir.glob("*.json"))
            stats[split]['total'] = len(json_files)
            stats[split]['original'] = len([f for f in json_files if 'original' in f.name])
            stats[split]['augmented'] = len([f for f in json_files if 'aug' in f.name])
    
    return stats

