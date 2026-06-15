"""
Data Augmentation Library for Training Data

Creates augmented versions of training images to increase dataset size
and improve model generalization with guaranteed diversity.

Enhanced Augmentation System (2026-01-12):
- Pre-shrink: 5% content shrink creates margins for transformations
- Rotation: ±2-5 degrees (excludes near-zero, progressive aggressiveness)
- Translation: ±3 pixels (excludes near-zero, proper magnitude increase)
- Local Warping: 15-20px displacement with IDW interpolation
- Diversity Control: SSIM-based filtering ensures unique augmentations
- Progressive Retry: Automatic parameter increase if too similar

All augmentations preserve:
- Binary images (black lines on white background)
- Line quality (2px normalized thickness)
- Aspect ratio (568×274)
- Image dimensions
"""

import numpy as np
import cv2
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional
import os
import json
import shutil
from pathlib import Path
from utils.logger import get_logger

# Import shared preprocessing functions
from .preprocessing import (
    apply_pre_shrink as _shared_apply_pre_shrink,
    apply_binarization,
    apply_line_normalization
)

logger = get_logger(__name__)

# Import SSIM for similarity checking
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    logger.warning("scikit-image not available, diversity control will be disabled")
    SSIM_AVAILABLE = False


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
    
    # Create displacement field using Inverse Distance Weighting (VECTORIZED)
    # This is ~100x faster than the pixel-by-pixel loop
    
    # Create meshgrid of all pixel coordinates
    y_coords, x_coords = np.meshgrid(np.arange(h, dtype=np.float32), 
                                      np.arange(w, dtype=np.float32), 
                                      indexing='ij')
    
    # Reshape to (h*w, 2) for vectorized distance calculation
    all_pixels = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1)  # Shape: (h*w, 2)
    
    # Calculate distances from all pixels to all control points
    # Broadcasting: (h*w, 1, 2) - (1, num_points, 2) = (h*w, num_points, 2)
    distances = np.linalg.norm(
        all_pixels[:, np.newaxis, :] - control_points[np.newaxis, :, :], 
        axis=2
    )  # Shape: (h*w, num_points)
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-6)
    
    # Inverse Distance Weighting: weight = 1 / distance^2
    weights = 1.0 / (distances ** 2)  # Shape: (h*w, num_points)
    weights = weights / np.sum(weights, axis=1, keepdims=True)  # Normalize
    
    # Calculate weighted displacements (vectorized)
    dx = np.sum(displacements[:, 0] * weights, axis=1)  # Shape: (h*w,)
    dy = np.sum(displacements[:, 1] * weights, axis=1)  # Shape: (h*w,)
    
    # Calculate edge distances for all pixels
    safety_margin_edge = 10
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    
    dist_to_left = x_flat
    dist_to_right = w - 1 - x_flat
    dist_to_top = y_flat
    dist_to_bottom = h - 1 - y_flat
    
    # Calculate edge reduction factors (vectorized)
    edge_factor_x = np.minimum(
        1.0,
        np.minimum(
            (dist_to_left - safety_margin_edge) / max(1, safety_margin_edge),
            (dist_to_right - safety_margin_edge) / max(1, safety_margin_edge)
        )
    )
    edge_factor_y = np.minimum(
        1.0,
        np.minimum(
            (dist_to_top - safety_margin_edge) / max(1, safety_margin_edge),
            (dist_to_bottom - safety_margin_edge) / max(1, safety_margin_edge)
        )
    )
    
    # Clamp to [0, 1]
    edge_factor_x = np.clip(edge_factor_x, 0.0, 1.0)
    edge_factor_y = np.clip(edge_factor_y, 0.0, 1.0)
    
    # Apply edge reduction
    dx = dx * edge_factor_x
    dy = dy * edge_factor_y
    
    # Calculate final mapping coordinates
    new_x = x_flat + dx
    new_y = y_flat + dy
    
    # Reshape back to (h, w) and clip to image bounds
    map_x = np.clip(new_x.reshape(h, w), 0, w - 1)
    map_y = np.clip(new_y.reshape(h, w), 0, h - 1)
    
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
    
    # Re-binarize and normalize line thickness using shared functions
    warped = apply_binarization(warped)
    warped = apply_line_normalization(warped)
    
    return warped


class ImageAugmentor:
    """
    Augments training images with realistic variations.
    
    NEW FEATURES:
    - Pre-shrink for margin creation
    - Diversity control (similarity-based filtering)
    - Pairwise uniqueness checking
    - Progressive aggressiveness on retry
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-5.0, 5.0),
        translation_range: Tuple[int, int] = (-3, 3),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        num_augmentations: int = 6,
        safety_margin: int = 15,
        # NEW: Pre-shrink settings (default mirrors config/training_config.yaml: 0.90)
        pre_shrink_enabled: bool = True,
        pre_shrink_factor: float = 0.90,
        # NEW: Diversity control settings
        diversity_control_enabled: bool = True,
        similarity_to_original_max: float = 0.95,
        similarity_between_augs_max: float = 0.93,
        max_attempts_per_augmentation: int = 10,
        progressive_aggressiveness: bool = True,
        # NEW: Progressive parameters
        rotation_multiplier_per_attempt: float = 0.15,
        translation_multiplier_per_attempt: float = 0.20,
        warping_displacement_increase: int = 3,
        max_rotation: float = 8.0,
        max_translation: int = 5,
        max_warping_displacement: int = 25,
        # NEW: Augmentation mix
        rotation_translation_ratio: float = 0.50,
        warping_only_ratio: float = 0.33,
        warping_combined_ratio: float = 0.17,
        # NEW: Warping settings
        warping_displacement_min: int = 15,
        warping_displacement_max: int = 20,
        # NEW: Exclude near-zero
        exclude_near_zero_rotation: float = 2.0,
        exclude_near_zero_translation: int = 1
    ):
        """
        Initialize augmentor with enhanced diversity control.
        
        Args:
            rotation_range: Min/max rotation in degrees
            translation_range: Min/max translation in pixels
            scale_range: Min/max scale factor
            num_augmentations: Number of augmented versions per image
            safety_margin: Minimum pixel margin from edges
            pre_shrink_enabled: Enable pre-shrink for margin creation
            pre_shrink_factor: Shrink factor (0.95 = 5% shrink)
            diversity_control_enabled: Enable similarity-based filtering
            similarity_to_original_max: Max SSIM to original (reject if higher)
            similarity_between_augs_max: Max SSIM between augmentations
            max_attempts_per_augmentation: Max retry attempts
            progressive_aggressiveness: Increase params on retry
            [Additional parameters documented in config.yaml]
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.num_augmentations = num_augmentations
        self.safety_margin = safety_margin
        
        # NEW: Pre-shrink settings
        self.pre_shrink_enabled = pre_shrink_enabled
        self.pre_shrink_factor = pre_shrink_factor
        
        # NEW: Diversity control
        self.diversity_control_enabled = diversity_control_enabled
        self.similarity_to_original_max = similarity_to_original_max
        self.similarity_between_augs_max = similarity_between_augs_max
        self.max_attempts_per_augmentation = max_attempts_per_augmentation
        self.progressive_aggressiveness = progressive_aggressiveness
        
        # NEW: Progressive parameters
        self.rotation_multiplier_per_attempt = rotation_multiplier_per_attempt
        self.translation_multiplier_per_attempt = translation_multiplier_per_attempt
        self.warping_displacement_increase = warping_displacement_increase
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.max_warping_displacement = max_warping_displacement
        
        # NEW: Augmentation mix
        self.rotation_translation_ratio = rotation_translation_ratio
        self.warping_only_ratio = warping_only_ratio
        self.warping_combined_ratio = warping_combined_ratio
        
        # NEW: Warping settings
        self.warping_displacement_min = warping_displacement_min
        self.warping_displacement_max = warping_displacement_max
        
        # NEW: Exclude near-zero
        self.exclude_near_zero_rotation = exclude_near_zero_rotation
        self.exclude_near_zero_translation = exclude_near_zero_translation
    
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
        # Ensure RGB format first (apply_binarization handles the conversion internally)
        if len(augmented.shape) == 2:
            augmented = cv2.cvtColor(augmented, cv2.COLOR_GRAY2RGB)
        
        # Use shared functions for consistency across all preprocessing paths
        augmented = apply_binarization(augmented)
        augmented = apply_line_normalization(augmented)
        
        return augmented
    
    def _apply_pre_shrink(self, image: np.ndarray) -> np.ndarray:
        """
        Apply pre-shrink to create margins for translation/rotation.
        
        Uses shared implementation from preprocessing.py for consistency.
        
        Args:
            image: Input image (H×W or H×W×3)
        
        Returns:
            Shrunk image centered on white canvas (same size as input)
        """
        if not self.pre_shrink_enabled or self.pre_shrink_factor >= 1.0:
            return image
        
        # Use shared implementation for consistency across all paths
        return _shared_apply_pre_shrink(image, factor=self.pre_shrink_factor)
    
    def _calculate_similarity(self, img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
        """
        Calculate structural similarity between two grayscale images.
        
        Args:
            img1_gray: First grayscale image
            img2_gray: Second grayscale image
        
        Returns:
            SSIM score (1.0 = identical, 0.0 = completely different)
        """
        if not SSIM_AVAILABLE:
            logger.warning("SSIM not available, returning 0.0 (assume different)")
            return 0.0
        
        try:
            similarity = ssim(img1_gray, img2_gray, data_range=255)
            return float(similarity)
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0
    
    def _get_augmentation_type(self, aug_idx: int, total_augs: int) -> str:
        """
        Determine augmentation type based on index and mix ratios.
        
        Args:
            aug_idx: Index of current augmentation (0-based)
            total_augs: Total number of augmentations
        
        Returns:
            Augmentation type: 'rotation_translation', 'warping_only', or 'warping_combined'
        """
        # Calculate boundaries
        rot_trans_count = int(total_augs * self.rotation_translation_ratio)
        warp_only_count = int(total_augs * self.warping_only_ratio)
        
        if aug_idx < rot_trans_count:
            return 'rotation_translation'
        elif aug_idx < rot_trans_count + warp_only_count:
            return 'warping_only'
        else:
            return 'warping_combined'
    
    def _generate_typed_augmentation(
        self,
        image: np.ndarray,
        aug_type: str,
        attempt: int = 1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate augmentation of specified type with progressive aggressiveness.
        
        Args:
            image: Input image (already pre-shrunk if enabled)
            aug_type: Type of augmentation
            attempt: Attempt number (1-based, increases aggressiveness)
        
        Returns:
            (augmented_image, parameters_dict)
        """
        # Calculate aggressiveness multiplier
        if self.progressive_aggressiveness and attempt > 1:
            aggr_mult = 1.0 + (attempt - 1) * self.rotation_multiplier_per_attempt
        else:
            aggr_mult = 1.0
        
        params = {'augmentation_type': aug_type, 'attempt': attempt}
        
        if aug_type == 'rotation_translation':
            # Generate rotation (exclude near-zero)
            if self.exclude_near_zero_rotation > 0:
                # Choose either negative or positive range
                if np.random.random() < 0.5:
                    rotation = np.random.uniform(
                        self.rotation_range[0],
                        -self.exclude_near_zero_rotation
                    )
                else:
                    rotation = np.random.uniform(
                        self.exclude_near_zero_rotation,
                        self.rotation_range[1]
                    )
            else:
                rotation = np.random.uniform(*self.rotation_range)
            
            # Apply aggressiveness multiplier
            rotation = rotation * aggr_mult
            rotation = np.clip(rotation, -self.max_rotation, self.max_rotation)
            
            # Generate translation (exclude near-zero if configured)
            if self.exclude_near_zero_translation > 0:
                # Choose either negative or positive range (exclude [-exclude_near_zero_translation, exclude_near_zero_translation])
                if np.random.random() < 0.5:
                    # Negative range: [translation_range[0], -exclude_near_zero_translation) (exclusive upper bound)
                    tx = np.random.randint(self.translation_range[0], -self.exclude_near_zero_translation)
                    ty = np.random.randint(self.translation_range[0], -self.exclude_near_zero_translation)
                else:
                    # Positive range: [exclude_near_zero_translation + 1, translation_range[1]] (inclusive)
                    tx = np.random.randint(self.exclude_near_zero_translation + 1, self.translation_range[1] + 1)
                    ty = np.random.randint(self.exclude_near_zero_translation + 1, self.translation_range[1] + 1)
            else:
                tx = np.random.randint(self.translation_range[0], self.translation_range[1] + 1)
                ty = np.random.randint(self.translation_range[0], self.translation_range[1] + 1)
            
            # Apply aggressiveness multiplier
            if self.progressive_aggressiveness and attempt > 1:
                trans_mult = 1.0 + (attempt - 1) * self.translation_multiplier_per_attempt
                # Use proper rounding to increase magnitude in both directions
                # int() truncates toward zero, so for negative values we need ceil
                tx = int(np.sign(tx) * np.ceil(np.abs(tx * trans_mult)))
                ty = int(np.sign(ty) * np.ceil(np.abs(ty * trans_mult)))
            
            tx = np.clip(tx, -self.max_translation, self.max_translation)
            ty = np.clip(ty, -self.max_translation, self.max_translation)
            
            # Safety check with fallback: Reduce rotation/translation if unsafe
            rotation_safe, tx_safe, ty_safe = rotation, tx, ty
            scale_safe = 1.0
            safety_check_applied = False
            
            # Try original parameters, then 50%, then 25%
            for reduction_factor in [1.0, 0.5, 0.25]:
                if reduction_factor < 1.0:
                    rotation_safe = rotation * reduction_factor
                    tx_safe = int(tx * reduction_factor)
                    ty_safe = int(ty * reduction_factor)
                    safety_check_applied = True
                
                if self._is_safe_augmentation(image, rotation_safe, tx_safe, ty_safe, scale_safe):
                    if safety_check_applied:
                        logger.debug(f"Safety fallback: rotation {rotation:.2f}°→{rotation_safe:.2f}°, "
                                   f"translation ({tx},{ty})→({tx_safe},{ty_safe})")
                    break
            else:
                # If all reductions fail, use minimal safe parameters
                rotation_safe = 0.0
                tx_safe = 0
                ty_safe = 0
                logger.warning(f"Extreme fallback: using minimal parameters (rotation=0°, translation=0px)")
                safety_check_applied = True
            
            # Apply augmentation with safe parameters
            aug_img = self.augment_image(image, rotation_safe, tx_safe, ty_safe, scale=scale_safe)
            
            params.update({
                'rotation': float(rotation_safe),
                'rotation_requested': float(rotation),
                'translation_x': int(tx_safe),
                'translation_x_requested': int(tx),
                'translation_y': int(ty_safe),
                'translation_y_requested': int(ty),
                'scale': 1.0,
                'safety_check_applied': safety_check_applied
            })
        
        elif aug_type == 'warping_only':
            # Variable warping displacement with aggressiveness
            base_displacement = np.random.randint(
                self.warping_displacement_min,
                self.warping_displacement_max + 1
            )
            
            if self.progressive_aggressiveness and attempt > 1:
                displacement = base_displacement + (attempt - 1) * self.warping_displacement_increase
            else:
                displacement = base_displacement
            
            displacement = min(displacement, self.max_warping_displacement)
            
            aug_img = apply_local_warp(
                image,
                num_control_points=9,
                max_displacement=displacement,
                safety_margin=15
            )
            
            params.update({
                'warping_displacement': int(displacement),
                'warping_control_points': 9
            })
        
        else:  # warping_combined
            # Step 1: Apply warping
            base_displacement = np.random.randint(
                self.warping_displacement_min,
                self.warping_displacement_max + 1
            )
            
            if self.progressive_aggressiveness and attempt > 1:
                displacement = base_displacement + (attempt - 1) * self.warping_displacement_increase
            else:
                displacement = base_displacement
            
            displacement = min(displacement, self.max_warping_displacement)
            
            warped = apply_local_warp(
                image,
                num_control_points=9,
                max_displacement=displacement,
                safety_margin=15
            )
            
            # Step 2: Apply light transformation (50% of normal range)
            rotation = np.random.uniform(*self.rotation_range) * 0.5 * aggr_mult
            rotation = np.clip(rotation, -self.max_rotation, self.max_rotation)
            
            # Generate translation with light range (-2 to 2), excluding near-zero if configured
            light_translation_min = -2
            light_translation_max = 2
            if self.exclude_near_zero_translation > 0:
                # Choose either negative or positive range (exclude [-exclude_near_zero_translation, exclude_near_zero_translation])
                if np.random.random() < 0.5:
                    # Negative range: [light_translation_min, -exclude_near_zero_translation - 1]
                    tx = np.random.randint(light_translation_min, -self.exclude_near_zero_translation)
                    ty = np.random.randint(light_translation_min, -self.exclude_near_zero_translation)
                else:
                    # Positive range: [exclude_near_zero_translation + 1, light_translation_max]
                    tx = np.random.randint(self.exclude_near_zero_translation + 1, light_translation_max + 1)
                    ty = np.random.randint(self.exclude_near_zero_translation + 1, light_translation_max + 1)
            else:
                tx = np.random.randint(light_translation_min, light_translation_max + 1)
                ty = np.random.randint(light_translation_min, light_translation_max + 1)
            
            if self.progressive_aggressiveness and attempt > 1:
                trans_mult = 1.0 + (attempt - 1) * self.translation_multiplier_per_attempt
                # Use proper rounding to increase magnitude in both directions
                # int() truncates toward zero, so for negative values we need ceil
                tx = int(np.sign(tx) * np.ceil(np.abs(tx * trans_mult)))
                ty = int(np.sign(ty) * np.ceil(np.abs(ty * trans_mult)))
            
            tx = np.clip(tx, -self.max_translation, self.max_translation)
            ty = np.clip(ty, -self.max_translation, self.max_translation)
            
            # Safety check with fallback: Reduce rotation/translation if unsafe
            rotation_safe, tx_safe, ty_safe = rotation, tx, ty
            scale_safe = 1.0
            safety_check_applied = False
            
            # Try original parameters, then 50%, then 25%
            for reduction_factor in [1.0, 0.5, 0.25]:
                if reduction_factor < 1.0:
                    rotation_safe = rotation * reduction_factor
                    tx_safe = int(tx * reduction_factor)
                    ty_safe = int(ty * reduction_factor)
                    safety_check_applied = True
                
                if self._is_safe_augmentation(warped, rotation_safe, tx_safe, ty_safe, scale_safe):
                    if safety_check_applied:
                        logger.debug(f"Safety fallback (warp+combined): rotation {rotation:.2f}°→{rotation_safe:.2f}°, "
                                   f"translation ({tx},{ty})→({tx_safe},{ty_safe})")
                    break
            else:
                # If all reductions fail, use minimal safe parameters
                rotation_safe = 0.0
                tx_safe = 0
                ty_safe = 0
                logger.warning(f"Extreme fallback (warp+combined): using minimal parameters")
                safety_check_applied = True
            
            # Apply augmentation with safe parameters
            aug_img = self.augment_image(warped, rotation_safe, tx_safe, ty_safe, scale=scale_safe)
            
            params.update({
                'warping_displacement': int(displacement),
                'warping_control_points': 9,
                'rotation': float(rotation_safe),
                'rotation_requested': float(rotation),
                'translation_x': int(tx_safe),
                'translation_x_requested': int(tx),
                'translation_y': int(ty_safe),
                'translation_y_requested': int(ty),
                'scale': 1.0,
                'safety_check_applied': safety_check_applied
            })
        
        return aug_img, params
    
    def _get_content_bounds(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box of actual content (non-white pixels).
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            (min_row, max_row, min_col, max_col) or None if no content
        """
        # Use higher threshold to avoid interpolation artifacts from cv2.resize()
        # After pre-shrink, resize creates gray pixels (anti-aliasing) at edges
        # Threshold of 200 filters these out, detecting only actual content
        if len(image.shape) == 3:
            # RGB: check if any channel < 200 (detect non-white content)
            content_mask = np.any(image < 200, axis=2)
        else:
            # Grayscale: check if < 200 (detect non-white content)
            content_mask = image < 200
        
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
        # Use more realistic rotation loss calculation for small angles
        # For small rotations (< 5°), corner displacement is minimal
        diagonal = np.sqrt(content_height**2 + content_width**2)
        rotation_margin_loss = int(diagonal * np.sin(rotation_rad) * 0.2)  # Further reduced for small angles
        
        # Translation directly reduces margins
        translation_margin_loss_x = abs(tx)
        translation_margin_loss_y = abs(ty)
        
        # Scaling up reduces effective margins
        scale_margin_loss = int(max(content_height, content_width) * (scale - 1.0) * 0.5) if scale > 1 else 0
        
        # Total margin requirements (minimal safety buffer for pre-shrunk images)
        # Pre-shrink already creates margins, safety margin can be minimal
        effective_safety_margin = 0  # No additional buffer needed with pre-shrink
        required_margin = effective_safety_margin + rotation_margin_loss + scale_margin_loss
        
        # Check if margins are sufficient (check direction-specific margins)
        safe_top = margin_top >= required_margin + translation_margin_loss_y if ty < 0 else margin_top >= required_margin
        safe_bottom = margin_bottom >= required_margin + translation_margin_loss_y if ty > 0 else margin_bottom >= required_margin
        safe_left = margin_left >= required_margin + translation_margin_loss_x if tx < 0 else margin_left >= required_margin
        safe_right = margin_right >= required_margin + translation_margin_loss_x if tx > 0 else margin_right >= required_margin
        
        is_safe = safe_top and safe_bottom and safe_left and safe_right
        
        # Debug logging for failed checks (only log once per image to avoid spam)
        if not is_safe and not hasattr(self, '_logged_safety_failure'):
            self._logged_safety_failure = True
            logger.debug(f"Safety check: rotation={rotation:.1f}°, tx={tx}, ty={ty}, "
                        f"margins=[{margin_top},{margin_bottom},{margin_left},{margin_right}], "
                        f"required={required_margin}, rot_loss={rotation_margin_loss}")
        
        return is_safe
    
    def augment_batch(
        self,
        image: np.ndarray,
        num_augmentations: int = None,
        use_warping: bool = True
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Create multiple augmented versions with pairwise diversity control.
        
        NEW ENHANCED STRATEGY:
        1. Pre-shrink image by 5% to create ~14px margins
        2. Generate augmentations with type-based distribution
        3. Check similarity to original (reject if >95% similar)
        4. Check similarity to all other augmentations (reject if >93% similar)
        5. Retry with progressively more aggressive parameters if rejected
        6. Track comprehensive diversity metrics
        
        Args:
            image: Input image
            num_augmentations: Number of augmentations (uses self.num_augmentations if None)
            use_warping: Enable warping (default: True)
        
        Returns:
            List of (augmented_image, parameters) tuples with diversity metrics
        """
        if num_augmentations is None:
            num_augmentations = self.num_augmentations
        
        # Step 1: Pre-shrink to create margins
        image_shrunk = self._apply_pre_shrink(image)
        
        # Convert to grayscale for similarity comparisons
        if len(image_shrunk.shape) == 3:
            original_gray = cv2.cvtColor(image_shrunk, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = image_shrunk.copy()
        
        augmented_images = []
        augmented_grays = []  # Store grayscale versions for pairwise comparison
        
        # Diversity statistics
        diversity_stats = {
            'accepted': 0,
            'rejected_vs_original': 0,
            'rejected_vs_augmentations': 0,
            'total_attempts': 0,
            'forced_accepts': 0,
            'similarity_to_original': [],
            'max_similarity_between_augs': []
        }
        
        # Check if diversity control should be used
        use_diversity_control = self.diversity_control_enabled and SSIM_AVAILABLE
        
        if not self.diversity_control_enabled:
            logger.warning("Diversity control is disabled, augmentations may be redundant")
        elif not SSIM_AVAILABLE:
            logger.warning("SSIM (scikit-image) not available - diversity control disabled, proceeding without similarity filtering")
            logger.warning("  Install scikit-image for diversity control: pip install scikit-image")
            use_diversity_control = False
        
        # Generate augmentations (with or without diversity control)
        for aug_idx in range(num_augmentations):
            attempts = 0
            accepted = False
            best_aug = None
            best_similarity_orig = 1.0
            
            aug_type = self._get_augmentation_type(aug_idx, num_augmentations)
            
            if use_diversity_control:
                # Path 1: Diversity control enabled - use similarity checking with retries
                while attempts < self.max_attempts_per_augmentation and not accepted:
                    attempts += 1
                    diversity_stats['total_attempts'] += 1
                    
                    # Generate augmentation with progressive aggressiveness
                    aug_img, params = self._generate_typed_augmentation(
                        image_shrunk,
                        aug_type,
                        attempt=attempts
                    )
                    
                    # Convert to grayscale for comparison
                    if len(aug_img.shape) == 3:
                        aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_RGB2GRAY)
                    else:
                        aug_gray = aug_img.copy()
                    
                    # Check 1: Similarity to original
                    sim_to_original = self._calculate_similarity(original_gray, aug_gray)
                    
                    if sim_to_original >= self.similarity_to_original_max:
                        diversity_stats['rejected_vs_original'] += 1
                        # Track best attempt
                        if sim_to_original < best_similarity_orig:
                            best_similarity_orig = sim_to_original
                            best_aug = (aug_img, params, aug_gray, sim_to_original)
                        continue  # Retry
                    
                    # Check 2: Similarity to all existing augmentations
                    max_sim_to_augs = 0.0
                    too_similar_to_aug = False
                    
                    for prev_gray in augmented_grays:
                        sim = self._calculate_similarity(prev_gray, aug_gray)
                        max_sim_to_augs = max(max_sim_to_augs, sim)
                        
                        if sim >= self.similarity_between_augs_max:
                            too_similar_to_aug = True
                            diversity_stats['rejected_vs_augmentations'] += 1
                            break
                    
                    if too_similar_to_aug:
                        # Track best attempt
                        if sim_to_original < best_similarity_orig:
                            best_similarity_orig = sim_to_original
                            best_aug = (aug_img, params, aug_gray, sim_to_original)
                        continue  # Retry
                    
                    # Passed both checks - accept!
                    augmented_images.append((aug_img, {
                        **params,
                        'similarity_to_original': float(sim_to_original),
                        'max_similarity_to_augmentations': float(max_sim_to_augs),
                        'attempts': attempts,
                        'quality': 'optimal',
                        'pre_shrink_applied': self.pre_shrink_enabled
                    }))
                    augmented_grays.append(aug_gray)
                    diversity_stats['accepted'] += 1
                    diversity_stats['similarity_to_original'].append(sim_to_original)
                    diversity_stats['max_similarity_between_augs'].append(max_sim_to_augs)
                    accepted = True
                
                if not accepted:
                    # Max attempts reached - check if best attempt is acceptable
                    if best_aug:
                        aug_img, params, aug_gray, sim = best_aug
                        
                        # CRITICAL: Discard if still too similar to original (safety conflict)
                        # This happens when safety fallback creates near-duplicates (e.g., 0° rotation)
                        # Better to have fewer augmentations than force-accept near-duplicates
                        if sim >= self.similarity_to_original_max:
                            logger.warning(f"Aug {aug_idx} ({aug_type}): Max attempts reached, "
                                          f"best similarity={sim:.3f} still exceeds threshold ({self.similarity_to_original_max:.3f})")
                            logger.warning(f"  → DISCARDING (likely due to safety fallback creating near-duplicate)")
                            diversity_stats['discarded_too_similar'] = diversity_stats.get('discarded_too_similar', 0) + 1
                        else:
                            # Best attempt is acceptable (below threshold)
                            logger.warning(f"Aug {aug_idx} ({aug_type}): Max attempts reached, "
                                          f"using best (sim_orig={sim:.3f})")
                            augmented_images.append((aug_img, {
                                **params,
                                'similarity_to_original': float(sim),
                                'max_similarity_to_augmentations': 0.0,
                                'attempts': attempts,
                                'quality': 'forced_accept',
                                'pre_shrink_applied': self.pre_shrink_enabled
                            }))
                            augmented_grays.append(aug_gray)
                            diversity_stats['forced_accepts'] += 1
                    else:
                        logger.error(f"Aug {aug_idx}: Failed to generate any augmentation")
            else:
                # Path 2: Diversity control disabled - generate augmentations directly without similarity checks
                attempts = 1
                diversity_stats['total_attempts'] += 1
                
                # Generate augmentation (use attempt=1 since no retries)
                aug_img, params = self._generate_typed_augmentation(
                    image_shrunk,
                    aug_type,
                    attempt=1
                )
                
                # Accept immediately (no similarity checks)
                augmented_images.append((aug_img, {
                    **params,
                    'similarity_to_original': None,
                    'max_similarity_to_augmentations': None,
                    'attempts': 1,
                    'quality': 'no_filtering',
                    'pre_shrink_applied': self.pre_shrink_enabled
                }))
                diversity_stats['accepted'] += 1
        
        # Calculate overall diversity metrics (only if diversity control was used)
        if use_diversity_control:
            avg_attempts = diversity_stats['total_attempts'] / num_augmentations if num_augmentations > 0 else 0
            avg_sim_to_orig = np.mean(diversity_stats['similarity_to_original']) if diversity_stats['similarity_to_original'] else 0
            avg_sim_between = np.mean(diversity_stats['max_similarity_between_augs']) if diversity_stats['max_similarity_between_augs'] else 0
            
            # Calculate diversity score (1.0 = completely different, 0.0 = identical)
            diversity_score = 1.0 - avg_sim_to_orig
            
            # Log comprehensive statistics
            logger.info(f"Augmentation diversity control:")
            logger.info(f"  Accepted: {diversity_stats['accepted']}/{num_augmentations}")
            logger.info(f"  Rejected (vs original): {diversity_stats['rejected_vs_original']}")
            logger.info(f"  Rejected (vs other augs): {diversity_stats['rejected_vs_augmentations']}")
            logger.info(f"  Forced accepts: {diversity_stats['forced_accepts']}")
            logger.info(f"  Avg attempts per aug: {avg_attempts:.1f}")
            logger.info(f"  Avg similarity to original: {avg_sim_to_orig:.3f}")
            logger.info(f"  Avg max similarity between augs: {avg_sim_between:.3f}")
            logger.info(f"  Diversity score: {diversity_score:.3f}")
        else:
            # Diversity control disabled - simplified logging
            logger.info(f"Augmentation (diversity control disabled):")
            logger.info(f"  Generated: {diversity_stats['accepted']}/{num_augmentations} augmentations")
            avg_attempts = 1.0  # Always 1 attempt when disabled
            avg_sim_to_orig = 0.0
            avg_sim_between = 0.0
            diversity_score = 0.0  # Not applicable
        
        # Add diversity summary to first augmentation's metadata
        if augmented_images:
            diversity_summary = {
                'diversity_control_enabled': use_diversity_control,
                'total_augmentations': num_augmentations,
                'accepted': diversity_stats['accepted'],
                'avg_attempts_per_aug': float(avg_attempts)
            }
            
            if use_diversity_control:
                diversity_summary.update({
                    'rejected_vs_original': diversity_stats['rejected_vs_original'],
                    'rejected_vs_augmentations': diversity_stats['rejected_vs_augmentations'],
                    'forced_accepts': diversity_stats['forced_accepts'],
                    'avg_similarity_to_original': float(avg_sim_to_orig),
                    'avg_max_similarity_between_augs': float(avg_sim_between),
                    'diversity_score': float(diversity_score)
                })
            
            augmented_images[0][1]['diversity_summary'] = diversity_summary
        
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
            logger.info(f"Cleaning existing augmented data in {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'train': {'original': 0, 'augmented': 0, 'total': 0},
            'val': {'original': 0, 'augmented': 0, 'total': 0},
            'errors': []
        }
        
        # Process training set
        logger.info(f"Augmenting training set ({len(split_indices['train'])} images)...")
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
        logger.info(f"Augmenting validation set ({len(split_indices['val'])} images)...")
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
                'num_augmentations': self.augmentor.num_augmentations,
                # Pre-shrink config (critical for prediction alignment)
                'pre_shrink': {
                    'enabled': self.augmentor.pre_shrink_enabled,
                    'factor': self.augmentor.pre_shrink_factor
                }
            },
            'include_original': self.include_original,
            'statistics': stats
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Expose the augmentation config (incl. pre_shrink) to the caller so it lands in
        # the MODEL metadata — inference reads pre_shrink/factor from there to reproduce
        # the exact training preprocessing.
        stats['augmentation_config'] = metadata['augmentation_config']

        logger.info(f"Augmented dataset created in {self.output_dir}")
        logger.info(f"Train: {stats['train']['total']} images ({stats['train']['original']} original + {stats['train']['augmented']} augmented)")
        logger.info(f"Val:   {stats['val']['total']} images ({stats['val']['original']} original + {stats['val']['augmented']} augmented)")
        
        if stats['errors']:
            logger.warning(f"{len(stats['errors'])} errors occurred during augmentation")
        
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

                # Check mode (Components sentinel / Custom_Class prefix / regression)
                from .dataset import COMPONENTS_TARGET, components_to_vector
                is_components = (target_feature == COMPONENTS_TARGET)
                is_classification = target_feature.startswith('Custom_Class_')
                target_value = None
                target_vector = None

                if is_components:
                    target_vector = components_to_vector(features)
                    if target_vector is None:
                        errors.append(f"Image {img_data.get('id', idx)} missing component sub-labels")
                        continue
                elif is_classification:
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

                # Apply normalization if normalizer is provided (regression only)
                target_value_normalized = target_value
                if self.normalizer is not None and target_value is not None:
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
                
                # Save original image (with pre-shrink for consistency)
                if self.include_original:
                    # Apply pre-shrink to original for consistency with augmented images
                    # This ensures all images in training dataset have same margins
                    original_to_save = self.augmentor._apply_pre_shrink(img_array)
                    
                    # Apply same post-processing as augmented images using shared functions
                    # Ensure RGB for consistent processing
                    if len(original_to_save.shape) == 2:
                        original_to_save = cv2.cvtColor(original_to_save, cv2.COLOR_GRAY2RGB)
                    
                    original_to_save = apply_binarization(original_to_save)
                    original_to_save = apply_line_normalization(original_to_save)
                    
                    # Convert to grayscale for saving (disk-optimized)
                    original_to_save = cv2.cvtColor(original_to_save, cv2.COLOR_RGB2GRAY)
                    
                    original_path = output_dir / f"{patient_id}_id{img_id}_original.png"
                    cv2.imwrite(str(original_path), original_to_save)
                    
                    # Save label (with normalized value if normalizer is used)
                    label_path = output_dir / f"{patient_id}_id{img_id}_original.json"
                    with open(label_path, 'w') as f:
                        json.dump({
                            'image_id': img_id,
                            'patient_id': patient_id,
                            'target_feature': target_feature,
                            'target_value': target_value_normalized,
                            'target_value_original': target_value,  # Keep original for reference
                            'target_vector': target_vector,  # 60-dim for component mode (else None)
                            'augmentation': None,
                            'pre_shrink_applied': self.augmentor.pre_shrink_enabled
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
                            'target_vector': target_vector,  # 60-dim for component mode (else None)
                            'augmentation': aug_params
                        }, f)
                    
                    stats['augmented'] += 1
                    stats['total'] += 1
                
                # Progress indicator
                if stats['total'] % 10 == 0:
                    logger.debug(f"Processed {stats['total']} images...")
                
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
            logger.warning(f"Image not found for {label_file.name}")
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
