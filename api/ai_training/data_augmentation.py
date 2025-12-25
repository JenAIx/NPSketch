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


def apply_local_warp(
    image: np.ndarray,
    num_control_points: int = 9,
    max_displacement: int = 15,
    safety_margin: int = 15,
    random_seed: int = None
) -> np.ndarray:
    """
    Apply local warping (TPS) to image for data augmentation.
    
    Args:
        image: Input image (H×W×3 or H×W)
        num_control_points: Number of control points (4 or 9)
        max_displacement: Maximum pixel displacement for control points
        safety_margin: Minimum distance from edge
        random_seed: Random seed for reproducibility
    
    Returns:
        Warped image
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    h, w = image.shape[:2]
    
    # Get control points in 3×3 grid at 25%, 50%, 75%
    control_points = []
    for v_pos in [0.25, 0.5, 0.75]:  # vertical
        for h_pos in [0.25, 0.5, 0.75]:  # horizontal
            control_points.append((w * h_pos, h * v_pos))
    control_points = np.array(control_points, dtype=np.float32)
    
    # Generate random displacements with border protection
    displaced_points = []
    for x, y in control_points:
        # Calculate distance to edges
        dist_to_left = x
        dist_to_right = w - x
        dist_to_top = y
        dist_to_bottom = h - y
        
        # Limit displacement based on distance to edge
        max_dx = min(max_displacement, dist_to_left - safety_margin, dist_to_right - safety_margin)
        max_dy = min(max_displacement, dist_to_top - safety_margin, dist_to_bottom - safety_margin)
        
        max_dx = max(0, max_dx)
        max_dy = max(0, max_dy)
        
        if max_dx < 2:
            max_dx = min(2, max_displacement * 0.3)
        if max_dy < 2:
            max_dy = min(2, max_displacement * 0.3)
        
        dx = np.random.randint(-int(max_dx), int(max_dx) + 1)
        dy = np.random.randint(-int(max_dy), int(max_dy) + 1)
        displaced_points.append((x + dx, y + dy))
    
    displaced_points = np.array(displaced_points, dtype=np.float32)
    
    # Calculate displacements
    displacements = displaced_points - control_points
    
    # Create displacement field using Inverse Distance Weighting
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    safety_margin_edge = 10
    
    for y in range(h):
        for x in range(w):
            pixel = np.array([x, y], dtype=np.float32)
            
            # Calculate distances to all control points
            distances = np.linalg.norm(control_points - pixel, axis=1)
            distances = np.maximum(distances, 1e-6)
            
            # Inverse Distance Weighting: weight = 1 / distance^2
            weights = 1.0 / (distances ** 2)
            weights = weights / np.sum(weights)
            
            # Calculate weighted displacement
            dx = np.sum(displacements[:, 0] * weights)
            dy = np.sum(displacements[:, 1] * weights)
            
            # Calculate distance to edges for edge reduction
            dist_to_left = x
            dist_to_right = w - 1 - x
            dist_to_top = y
            dist_to_bottom = h - 1 - y
            
            # Reduce displacement near edges
            edge_factor_x = min(1.0, 
                               (dist_to_left - safety_margin_edge) / max(1, safety_margin_edge),
                               (dist_to_right - safety_margin_edge) / max(1, safety_margin_edge))
            edge_factor_y = min(1.0,
                               (dist_to_top - safety_margin_edge) / max(1, safety_margin_edge),
                               (dist_to_bottom - safety_margin_edge) / max(1, safety_margin_edge))
            
            edge_factor_x = max(0.0, min(1.0, edge_factor_x))
            edge_factor_y = max(0.0, min(1.0, edge_factor_y))
            
            dx = dx * edge_factor_x
            dy = dy * edge_factor_y
            
            # Calculate final mapping coordinates
            new_x = x + dx
            new_y = y + dy
            
            map_x[y, x] = np.clip(new_x, 0, w - 1)
            map_y[y, x] = np.clip(new_y, 0, h - 1)
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image.copy()
    
    # Apply warping
    border_value = (255, 255, 255) if len(image_rgb.shape) == 3 else 255
    warped = cv2.remap(
        image_rgb, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    
    # Ensure output is RGB
    if len(warped.shape) == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
    
    # Re-binarize and normalize line thickness
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    warped = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    from line_normalizer import normalize_line_thickness
    warped = normalize_line_thickness(warped, target_thickness=2.0)
    
    return warped


class ImageAugmentor:
    """
    Augments training images with realistic variations.
    
    Includes content-aware bounds protection to prevent clipping lines.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-3.0, 3.0),
        translation_range: Tuple[int, int] = (-10, 10),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        num_augmentations: int = 5,
        safety_margin: int = 15
    ):
        """
        Initialize augmentor.
        
        Args:
            rotation_range: Min/max rotation in degrees
            translation_range: Min/max translation in pixels
            scale_range: Min/max scale factor
            num_augmentations: Number of augmented versions per image
            safety_margin: Minimum pixel margin from edges (default: 15)
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.num_augmentations = num_augmentations
        self.safety_margin = safety_margin
    
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
        
        # Determine border value based on image type
        # Medical drawings have white backgrounds (255)
        if len(image.shape) == 3:
            border_value = (255, 255, 255)  # RGB white
        else:
            border_value = 255  # Grayscale white
        
        # Apply transformation
        # Use INTER_LINEAR for smooth edges and white background to match original images
        augmented = cv2.warpAffine(
            image,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )
        
        # Re-binarize to ensure consistent binary images (no grayscale from interpolation)
        # Use threshold 175: optimal balance between line preservation and anti-fragmentation
        if len(augmented.shape) == 3:
            # RGB image: convert to grayscale, binarize, convert back to RGB
            gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            augmented = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        else:
            # Grayscale image: binarize directly
            _, augmented = cv2.threshold(augmented, 175, 255, cv2.THRESH_BINARY)
        
        # Re-normalize line thickness after augmentation to ensure consistent 2px lines
        # This is CRITICAL: Augmentation changes line thickness through interpolation
        # Skeleton + Dilation brings it back to consistent 2px
        from line_normalizer import normalize_line_thickness
        augmented = normalize_line_thickness(augmented, target_thickness=2.0)
        
        return augmented
    
    def _get_content_bounds(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of actual content (non-black pixels).
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            (min_row, max_row, min_col, max_col) or None if no content
        """
        # Threshold to detect non-background pixels
        if len(image.shape) == 3:
            # RGB: check if any channel > 10
            content_mask = np.any(image > 10, axis=2)
        else:
            # Grayscale: check if > 10
            content_mask = image > 10
        
        rows, cols = np.where(content_mask)
        
        if len(rows) == 0:
            # No content found
            return None
        
        return rows.min(), rows.max(), cols.min(), cols.max()
    
    def _is_safe_augmentation(
        self,
        image: np.ndarray,
        rotation: float,
        tx: int,
        ty: int,
        scale: float
    ) -> bool:
        """
        Check if augmentation parameters are safe (won't clip content).
        
        Args:
            image: Input image
            rotation: Rotation angle in degrees
            tx: Translation X
            ty: Translation Y
            scale: Scale factor
        
        Returns:
            True if safe, False if content might be clipped
        """
        bounds = self._get_content_bounds(image)
        
        if bounds is None:
            # No content, safe to augment
            return True
        
        min_row, max_row, min_col, max_col = bounds
        height, width = image.shape[:2]
        
        # Calculate content dimensions (inclusive bounds, so +1)
        content_height = max_row - min_row + 1
        content_width = max_col - min_col + 1
        
        # Check current margins
        # margin_top: pixels before content (0 to min_row-1)
        margin_top = min_row
        # margin_bottom: pixels after content (max_row+1 to height-1)
        margin_bottom = height - max_row - 1
        # margin_left: pixels before content (0 to min_col-1)
        margin_left = min_col
        # margin_right: pixels after content (max_col+1 to width-1)
        margin_right = width - max_col - 1
        
        # Calculate worst-case margin loss from transformations
        # Rotation can cause corners to extend further
        max_dimension = max(content_height, content_width)
        rotation_rad = abs(rotation) * np.pi / 180
        rotation_margin_loss = int(max_dimension * np.sin(rotation_rad) * 0.5)
        
        # Translation directly reduces margins
        translation_margin_loss_x = abs(tx)
        translation_margin_loss_y = abs(ty)
        
        # Scaling up reduces effective margins
        scale_margin_loss = int(max(content_height, content_width) * (scale - 1.0) * 0.5) if scale > 1 else 0
        
        # Total margin requirements
        required_margin = self.safety_margin + rotation_margin_loss + scale_margin_loss
        
        # Check if margins are sufficient
        safe_top = margin_top >= required_margin + translation_margin_loss_y if ty < 0 else margin_top >= required_margin
        safe_bottom = margin_bottom >= required_margin + translation_margin_loss_y if ty > 0 else margin_bottom >= required_margin
        safe_left = margin_left >= required_margin + translation_margin_loss_x if tx < 0 else margin_left >= required_margin
        safe_right = margin_right >= required_margin + translation_margin_loss_x if tx > 0 else margin_right >= required_margin
        
        return safe_top and safe_bottom and safe_left and safe_right
    
    def augment_batch(
        self,
        image: np.ndarray,
        num_augmentations: int = None,
        use_warping: bool = True
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Create multiple augmented versions of an image with content protection.
        
        NEW STRATEGY (Option 1): Combines global and local augmentations:
        - First 60% (3/5): Pure global augmentation (rotation, translation, scaling)
        - Last 40% (2/5): Local warping + global augmentation
        
        This provides diverse transformations while keeping the same number of images.
        
        Args:
            image: Input image
            num_augmentations: Number of augmentations (uses self.num_augmentations if None)
            use_warping: Enable warping for subset of augmentations (default: True)
        
        Returns:
            List of (augmented_image, parameters) tuples with augmentation info
        """
        if num_augmentations is None:
            num_augmentations = self.num_augmentations
        
        augmented_images = []
        safety_stats = {'safe': 0, 'conservative': 0}
        
        # Calculate split: 60% global, 40% warp+global
        num_global_only = int(num_augmentations * 0.6)
        num_warp_global = num_augmentations - num_global_only
        
        # Generate pure global augmentations
        for i in range(num_global_only):
            # Generate random parameters
            rotation = np.random.uniform(*self.rotation_range)
            tx = np.random.randint(*self.translation_range)
            ty = np.random.randint(*self.translation_range)
            scale = np.random.uniform(*self.scale_range)
            
            # Check if safe
            is_safe = self._is_safe_augmentation(image, rotation, tx, ty, scale)
            
            if not is_safe:
                # Use conservative parameters (50% reduction)
                rotation = rotation * 0.5
                tx = int(tx * 0.5)
                ty = int(ty * 0.5)
                
                # Verify conservative params are safe
                if not self._is_safe_augmentation(image, rotation, tx, ty, scale):
                    # Even more conservative: minimal transformation
                    rotation = rotation * 0.5
                    tx = int(tx * 0.5)
                    ty = int(ty * 0.5)
                    scale = 1.0 + (scale - 1.0) * 0.5
                
                safety_stats['conservative'] += 1
            else:
                safety_stats['safe'] += 1
            
            # Apply global augmentation only
            aug_img = self.augment_image(image, rotation, tx, ty, scale)
            
            params = {
                'augmentation_type': 'global',
                'rotation': float(rotation),
                'translation_x': int(tx),
                'translation_y': int(ty),
                'scale': float(scale),
                'safety_adjusted': not is_safe
            }
            
            augmented_images.append((aug_img, params))
        
        # Generate warp + global augmentations
        if use_warping:
            for i in range(num_warp_global):
                # Step 1: Apply local warping
                warped = apply_local_warp(
                    image,
                    num_control_points=9,
                    max_displacement=15,
                    safety_margin=15,
                    random_seed=None  # Random each time
                )
                
                # Step 2: Apply global augmentation on warped image
                rotation = np.random.uniform(*self.rotation_range)
                tx = np.random.randint(*self.translation_range)
                ty = np.random.randint(*self.translation_range)
                scale = np.random.uniform(*self.scale_range)
                
                # Check if safe (on warped image)
                is_safe = self._is_safe_augmentation(warped, rotation, tx, ty, scale)
                
                if not is_safe:
                    rotation = rotation * 0.5
                    tx = int(tx * 0.5)
                    ty = int(ty * 0.5)
                    
                    if not self._is_safe_augmentation(warped, rotation, tx, ty, scale):
                        rotation = rotation * 0.5
                        tx = int(tx * 0.5)
                        ty = int(ty * 0.5)
                        scale = 1.0 + (scale - 1.0) * 0.5
                    
                    safety_stats['conservative'] += 1
                else:
                    safety_stats['safe'] += 1
                
                # Apply global augmentation
                aug_img = self.augment_image(warped, rotation, tx, ty, scale)
                
                params = {
                    'augmentation_type': 'warp+global',
                    'warp_control_points': 9,
                    'warp_max_displacement': 15,
                    'rotation': float(rotation),
                    'translation_x': int(tx),
                    'translation_y': int(ty),
                    'scale': float(scale),
                    'safety_adjusted': not is_safe
                }
                
                augmented_images.append((aug_img, params))
        else:
            # If warping disabled, fill remaining with global augmentations
            for i in range(num_warp_global):
                rotation = np.random.uniform(*self.rotation_range)
                tx = np.random.randint(*self.translation_range)
                ty = np.random.randint(*self.translation_range)
                scale = np.random.uniform(*self.scale_range)
                
                is_safe = self._is_safe_augmentation(image, rotation, tx, ty, scale)
                
                if not is_safe:
                    rotation = rotation * 0.5
                    tx = int(tx * 0.5)
                    ty = int(ty * 0.5)
                    
                    if not self._is_safe_augmentation(image, rotation, tx, ty, scale):
                        rotation = rotation * 0.5
                        tx = int(tx * 0.5)
                        ty = int(ty * 0.5)
                        scale = 1.0 + (scale - 1.0) * 0.5
                    
                    safety_stats['conservative'] += 1
                else:
                    safety_stats['safe'] += 1
                
                aug_img = self.augment_image(image, rotation, tx, ty, scale)
                
                params = {
                    'augmentation_type': 'global',
                    'rotation': float(rotation),
                    'translation_x': int(tx),
                    'translation_y': int(ty),
                    'scale': float(scale),
                    'safety_adjusted': not is_safe
                }
                
                augmented_images.append((aug_img, params))
        
        # Log safety statistics
        if safety_stats['conservative'] > 0:
            print(f"  ⚠️ Content protection: {safety_stats['conservative']}/{num_augmentations} augmentations used conservative parameters")
        
        return augmented_images


class AugmentedDatasetBuilder:
    """
    Builds augmented training/test datasets and saves to disk.
    """
    
    def __init__(
        self,
        output_dir: str,
        augmentor: ImageAugmentor = None,
        include_original: bool = True,
        normalizer=None
    ):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save augmented data
            augmentor: ImageAugmentor instance (creates default if None)
            include_original: Whether to include original images
            normalizer: Optional TargetNormalizer for target values
        """
        self.output_dir = Path(output_dir)
        self.augmentor = augmentor or ImageAugmentor()
        self.include_original = include_original
        self.normalizer = normalizer
    
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
                
                # Check if feature exists (handle Custom_Class)
                is_classification = target_feature.startswith('Custom_Class_')
                target_value = None
                
                if is_classification:
                    num_classes_str = target_feature.replace('Custom_Class_', '')
                    if "Custom_Class" in features and num_classes_str in features.get("Custom_Class", {}):
                        target_value = features["Custom_Class"][num_classes_str]["label"]
                    else:
                        errors.append(f"Image {img_data.get('id', idx)} missing Custom_Class[{num_classes_str}]")
                        continue
                else:
                    if target_feature not in features:
                        errors.append(f"Image {img_data.get('id', idx)} missing feature {target_feature}")
                        continue
                    target_value = features[target_feature]
                
                # Apply normalization if normalizer is provided
                target_value_normalized = target_value
                if self.normalizer is not None:
                    target_value_normalized = self.normalizer.transform(np.array([target_value]))[0]
                
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
                    
                    # Save label (with normalized value if normalizer is used)
                    label_path = output_dir / f"{patient_id}_id{img_id}_original.json"
                    with open(label_path, 'w') as f:
                        json.dump({
                            'image_id': img_id,
                            'patient_id': patient_id,
                            'target_feature': target_feature,
                            'target_value': target_value_normalized,
                            'target_value_original': target_value,  # Keep original for reference
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
                    
                    # Save label with augmentation parameters (with normalized value if normalizer is used)
                    label_path = output_dir / f"{patient_id}_id{img_id}_aug{aug_idx}.json"
                    with open(label_path, 'w') as f:
                        json.dump({
                            'image_id': img_id,
                            'patient_id': patient_id,
                            'target_feature': target_feature,
                            'target_value': target_value_normalized,
                            'target_value_original': target_value,  # Keep original for reference
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

