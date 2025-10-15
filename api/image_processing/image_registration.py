"""
Image registration module for aligning drawings with references.

This module uses scikit-image's professional registration algorithms
specifically designed for image alignment tasks.

Key Design Principles:
1. Convert to binary (threshold) - focus on lines only
2. Phase cross-correlation for sub-pixel translation accuracy
3. Exhaustive rotation search with IoU scoring
4. SimilarityTransform for smooth, high-quality warping
5. Professional, peer-reviewed algorithms from scikit-image
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from skimage import transform as tf
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, SimilarityTransform
from scipy import ndimage
from scipy.optimize import minimize, differential_evolution


class ImageRegistration:
    """
    Handles image registration for aligning simple line drawings.
    
    Supports: Translation (X, Y) + Rotation + Uniform Zoom
    Method: Feature-based matching on edge maps
    """
    
    def __init__(self):
        """Initialize the image registration."""
        pass
    
    def register_images(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        method: str = "ecc",
        motion_type: str = "similarity",
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register source image to reference image using scikit-image.
        
        PROFESSIONAL IMPLEMENTATION using scikit-image:
        - Phase Cross-Correlation for translation (sub-pixel accuracy!)
        - Iterative rotation search with IoU scoring
        - SimilarityTransform for final warping
        
        Args:
            source: Source image to be aligned (uploaded drawing)
            reference: Reference image to align to
            method: Registration method (ignored, always uses scikit-image)
            motion_type: Type of motion (ignored, always similarity)
            max_rotation_degrees: Maximum allowed rotation in degrees
            
        Returns:
            Tuple of (registered_image, registration_info)
        """
        print(f"🎯 Professional Registration with scikit-image (max_rot={max_rotation_degrees}°)")
        
        # Use scikit-image's professional algorithms!
        return self._register_with_skimage(source, reference, max_rotation_degrees)
    
    def _register_with_skimage(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        ADVANCED OPTIMIZATION registration using scipy.optimize.
        
        Strategy:
        1. Binarization (threshold=127 for clear black/white)
        2. Use Differential Evolution - global optimizer
        3. Optimize rotation + scale + translation simultaneously
        4. IoU (Intersection over Union) as objective function
        5. Apply best transformation
        
        Much faster than brute force and finds optimal solution!
        
        Args:
            source: Source drawing
            reference: Reference drawing
            max_rotation_degrees: Maximum rotation to try
            
        Returns:
            Registered image and transformation info
        """
        # Convert to grayscale
        if len(source.shape) == 3:
            src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            src_gray = source.copy()
            
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference.copy()
        
        # Resize if needed
        if src_gray.shape != ref_gray.shape:
            src_gray = cv2.resize(src_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        # STRONG binary threshold (127 = clear black/white)
        print(f"  📐 Binary threshold (0-255) → Black/White")
        _, src_binary = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY_INV)
        _, ref_binary = cv2.threshold(ref_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Count pixels
        src_pixels = np.sum(src_binary > 0)
        ref_pixels = np.sum(ref_binary > 0)
        print(f"  📊 Black pixels: source={src_pixels}, reference={ref_pixels}")
        
        if src_pixels < 100 or ref_pixels < 100:
            print(f"  ⚠️  Too few pixels, using original")
            return source.copy(), {
                'method': 'simple_brute_force',
                'success': False,
                'reason': 'Too few black pixels',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        # PRE-SCALE: Normalize drawing size based on bounding box (height AND width)
        # BUT: Skip if images are already same size (pre-scaled by caller)
        initial_scale = 1.0
        
        # Find bounding boxes
        src_coords = np.column_stack(np.where(src_binary > 0))
        ref_coords = np.column_stack(np.where(ref_binary > 0))
        
        if len(src_coords) > 0 and len(ref_coords) > 0:
            # Get bounding box dimensions (height and width)
            src_height = src_coords[:, 0].max() - src_coords[:, 0].min()
            src_width = src_coords[:, 1].max() - src_coords[:, 1].min()
            ref_height = ref_coords[:, 0].max() - ref_coords[:, 0].min()
            ref_width = ref_coords[:, 1].max() - ref_coords[:, 1].min()
            
            # Calculate scale for both dimensions
            scale_h = ref_height / src_height if src_height > 0 else 1.0
            scale_w = ref_width / src_width if src_width > 0 else 1.0
            
            # Use MINIMUM to ensure we don't exceed canvas in either dimension
            potential_scale = min(scale_h, scale_w)
            
            print(f"  🔍 Scale check: src={src_height}x{src_width}px, ref={ref_height}x{ref_width}px")
            print(f"     → scale_h={scale_h:.2f}x, scale_w={scale_w:.2f}x → potential scale={potential_scale:.2f}x")
            
            # SKIP pre-scaling if already similar size (between 0.85 and 1.15)
            if 0.85 <= potential_scale <= 1.15:
                print(f"  ✓ Images already similar size, skipping pre-scaling")
                initial_scale = 1.0
            else:
                # Apply pre-scaling only if needed
                initial_scale = np.clip(potential_scale, 0.5, 2.5)
                print(f"  🔄 Applying pre-scale: {initial_scale:.2f}x")
                
                # Apply initial scale to BOTH source and src_gray
                h_scaled = int(src_gray.shape[0] * initial_scale)
                w_scaled = int(src_gray.shape[1] * initial_scale)
                
                # Scale the color/original source image
                source = cv2.resize(source, (w_scaled, h_scaled), interpolation=cv2.INTER_LINEAR)
                src_gray = cv2.resize(src_gray, (w_scaled, h_scaled), interpolation=cv2.INTER_LINEAR)
                
                # Re-threshold after scaling
                _, src_binary = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY_INV)
                src_pixels = np.sum(src_binary > 0)
                
                # Center on target canvas instead of resizing (prevents clipping!)
                if src_gray.shape != ref_gray.shape:
                    # Create white canvas with target size
                    target_h, target_w = ref_gray.shape
                    
                    # Center the scaled image on canvas
                    source = self._center_on_canvas(source, (target_h, target_w), value=255)
                    src_gray = self._center_on_canvas(src_gray, (target_h, target_w), value=255)
                    
                    _, src_binary = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY_INV)
                    src_pixels = np.sum(src_binary > 0)
                
                print(f"  ✓ After pre-scaling & centering: source={src_pixels} pixels")
        
        print(f"  🎯 SCIPY OPTIMIZATION: Differential Evolution (global optimizer)")
        print(f"     Search space: rotation ±{max_rotation_degrees}°, scale 0.7-1.4x, translation ±20px")
        
        h, w = src_binary.shape
        center = np.array([w / 2, h / 2])
        
        # Define objective function (returns NEGATIVE IoU for minimization)
        eval_count = [0]  # Mutable counter for function evaluations
        
        def objective(params):
            """Objective function: negative IoU for minimization"""
            angle_deg, scale, tx, ty = params
            eval_count[0] += 1
            
            # Create SimilarityTransform
            tform = SimilarityTransform(
                scale=scale,
                rotation=np.deg2rad(angle_deg),
                translation=(0, 0)
            )
            
            # Adjust for rotation/scale around center
            tform_center = SimilarityTransform(translation=-center)
            tform_uncenter = SimilarityTransform(translation=center)
            tform_combined = tform_center + tform + tform_uncenter
            
            # Apply transformation
            try:
                transformed = warp(
                    src_binary,
                    tform_combined.inverse,
                    output_shape=src_binary.shape,
                    order=1,
                    mode='constant',
                    cval=0,
                    preserve_range=True
                ).astype(np.uint8)
                
                # Apply translation
                M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
                translated = cv2.warpAffine(transformed, M_trans, (w, h),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)
                
                # Calculate IoU
                intersection = np.sum((translated > 127) & (ref_binary > 127))
                union = np.sum((translated > 127) | (ref_binary > 127))
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 0
                
                # Log progress every 50 evaluations
                if eval_count[0] % 50 == 0:
                    print(f"     Eval {eval_count[0]:4d}: angle={angle_deg:+6.2f}°, scale={scale:.3f}x, IoU={iou:.4f}")
                
                return -iou  # Negative for minimization
                
            except Exception as e:
                print(f"     ⚠️  Eval error: {e}")
                return 0  # Worst score
        
        # Define parameter bounds: [angle, scale, tx, ty]
        bounds = [
            (-max_rotation_degrees, max_rotation_degrees),  # Rotation (degrees)
            (0.7, 1.4),   # Scale
            (-20, 20),    # Translation X (pixels)
            (-20, 20)     # Translation Y (pixels)
        ]
        
        # Run Differential Evolution optimizer
        print(f"     🚀 Starting optimization...")
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',  # Good for noisy functions
            maxiter=80,           # Max generations
            popsize=12,           # Population size (12*4=48 individuals)
            tol=0.0001,          # Convergence tolerance
            atol=0.0001,         # Absolute tolerance
            seed=42,              # Reproducible
            workers=1,            # Single-threaded
            updating='immediate', # Faster convergence
            polish=True,          # Local refinement (L-BFGS-B)
            disp=False            # No verbose output
        )
        
        # Extract best parameters
        best_angle, best_scale, best_tx, best_ty = result.x
        best_score = -result.fun  # Negate back to positive IoU
        
        print(f"  ✅ Best: angle={best_angle:.2f}°, scale={best_scale:.3f}x, tx={best_tx:.1f}, ty={best_ty:.1f}, IoU={best_score:.4f}")
        
        # LOWERED threshold from 0.12 to 0.05 for better acceptance
        if best_score < 0.05:
            print(f"  ⚠️  Very low IoU ({best_score:.4f} < 0.05), using original")
            return source.copy(), {
                'method': 'scipy_differential_evolution',
                'success': False,
                'reason': f'Very low IoU score ({best_score:.4f} < 0.05)',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0,
                'overlap_score': float(best_score)
            }
        
        # Apply transformation to COLOR image
        print(f"  🎨 Applying transformation to color image...")
        
        # Step 1: Apply Scale + Rotation around center using SimilarityTransform
        center_point = np.array([w / 2, h / 2])
        
        # Create transformation WITHOUT translation first
        tform_scale_rot = SimilarityTransform(
            scale=best_scale,
            rotation=np.deg2rad(best_angle),
            translation=(0, 0)  # No translation yet!
        )
        
        # Adjust for rotation/scale around center
        tform_center = SimilarityTransform(translation=-center_point)
        tform_uncenter = SimilarityTransform(translation=center_point)
        tform_combined = tform_center + tform_scale_rot + tform_uncenter
        
        # Warp the color image (scale + rotation)
        if len(source.shape) == 3:
            temp_registered = np.zeros_like(source)
            for i in range(3):
                temp_registered[:, :, i] = warp(
                    source[:, :, i],
                    tform_combined.inverse,
                    output_shape=source.shape[:2],
                    order=1,
                    mode='constant',
                    cval=255,
                    preserve_range=True
                )
            temp_registered = temp_registered.astype(np.uint8)
        else:
            temp_registered = warp(
                source,
                tform_combined.inverse,
                output_shape=source.shape,
                order=1,
                mode='constant',
                cval=255,
                preserve_range=True
            ).astype(np.uint8)
        
        # Step 2: Apply Translation separately
        if best_tx != 0 or best_ty != 0:
            M_trans = np.float32([[1, 0, best_tx], [0, 1, best_ty]])
            registered = cv2.warpAffine(
                temp_registered,
                M_trans,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
            )
        else:
            registered = temp_registered
        
        # Clean borders
        registered = self._clean_borders(registered, border_size=5)
        
        # Thin lines if scaling made them thicker
        total_scale = initial_scale * best_scale
        if total_scale > 1.1:
            print(f"  ✂️ Thinning lines (scale={total_scale:.2f}x)...")
            registered = self._thin_lines(registered)
        
        print(f"  🎉 Registration successful! Final transform applied ✓")
        
        return registered, {
            'method': 'scipy_differential_evolution',
            'success': True,
            'translation_x': float(best_tx),
            'translation_y': float(best_ty),
            'rotation_degrees': float(best_angle),
            'scale': float(total_scale),  # Combined scale
            'overlap_score': float(best_score),
            'initial_scale': float(initial_scale)
        }
    
    def _register_exhaustive_search_OLD(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register using exhaustive rotation search with template matching.
        
        This is THE solution for simple line drawings!
        
        Strategy:
        1. Try discrete rotations (-30° to +30° in 3° steps)
        2. For each rotation, use normalized cross-correlation
        3. Pick the rotation with highest correlation
        4. Apply transformation
        
        Why this works:
        - No feature matching needed!
        - Correlation measures pixel-level similarity
        - Robust for repetitive structures
        - Fast enough (only ~20 rotations to try)
        - Predictable and debuggable
        
        Args:
            source: Source drawing
            reference: Reference drawing
            max_rotation_degrees: Maximum rotation to try
            
        Returns:
            Registered image and transformation info
        """
        # Convert to grayscale
        if len(source.shape) == 3:
            src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            src_gray = source.copy()
            
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference.copy()
        
        # Resize if needed
        if src_gray.shape != ref_gray.shape:
            src_gray = cv2.resize(src_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        # Convert to BINARY edge maps (only black lines matter!)
        # This is KEY for line drawings with lots of white space
        print(f"  📐 Converting to binary edge maps...")
        _, src_binary = cv2.threshold(src_gray, 200, 255, cv2.THRESH_BINARY_INV)
        _, ref_binary = cv2.threshold(ref_gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Now we have: Black pixels = lines, White pixels = background
        
        print(f"  🔄 Trying rotations from -{max_rotation_degrees}° to +{max_rotation_degrees}° (step=3°)")
        
        # Generate rotation angles to try (every 3 degrees)
        angles = list(range(0, int(max_rotation_degrees) + 1, 3))
        angles += [-a for a in angles if a != 0]
        angles = sorted(angles)
        
        best_score = -1
        best_angle = 0
        best_matrix = None
        
        center = (src_binary.shape[1] / 2, src_binary.shape[0] / 2)
        
        # Try each rotation
        for angle in angles:
            # Create rotation matrix (no scaling)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate source binary
            rotated = cv2.warpAffine(
                src_binary,
                M,
                (src_binary.shape[1], src_binary.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0  # Black background for binary
            )
            
            # Calculate IoU (Intersection over Union) of black pixels
            # This is MUCH better for sparse binary images!
            intersection = np.logical_and(rotated > 127, ref_binary > 127).sum()
            union = np.logical_or(rotated > 127, ref_binary > 127).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0
            
            if angle % 6 == 0:  # Log every 6 degrees
                print(f"    {angle:+4d}° → IoU={iou:.4f}")
            
            if iou > best_score:
                best_score = iou
                best_angle = angle
                best_matrix = M.copy()
        
        print(f"  ✅ Best rotation: {best_angle}° (IoU={best_score:.4f})")
        
        if best_matrix is None or best_score < 0.1:
            print(f"  ⚠️  Low IoU ({best_score:.4f} < 0.1), using original")
            return source.copy(), {
                'method': 'exhaustive_search',
                'success': False,
                'reason': f'Low IoU ({best_score:.4f})',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0,
                'iou': float(best_score)
            }
        
        # Apply best rotation to color image
        registered = cv2.warpAffine(
            source,
            best_matrix,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
        )
        
        # Clean borders
        registered = self._clean_borders(registered, border_size=5)
        
        print(f"  🎉 Registration successful!")
        
        return registered, {
            'method': 'exhaustive_search',
            'success': True,
            'translation_x': float(best_matrix[0, 2]),
            'translation_y': float(best_matrix[1, 2]),
            'rotation_degrees': float(best_angle),
            'scale': 1.0,
            'iou': float(best_score)
        }
    
    def _register_line_drawing_OLD(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register line drawing using edge-based feature matching.
        
        Strategy:
        1. Detect edges (Canny) - lines ARE edges!
        2. Find ORB features on edge maps
        3. Match features with RANSAC
        4. Estimate Similarity Transform (X/Y + Rot + Zoom)
        5. Validate and apply
        
        Args:
            source: Source drawing
            reference: Reference drawing
            max_rotation_degrees: Maximum rotation allowed
            
        Returns:
            Registered image and transformation info
        """
        # Convert to grayscale
        if len(source.shape) == 3:
            src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            src_gray = source.copy()
            
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference.copy()
        
        # Resize if needed
        if src_gray.shape != ref_gray.shape:
            src_gray = cv2.resize(src_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        print(f"  📐 Step 1: Edge detection...")
        # Edge detection - this is KEY for line drawings!
        src_edges = cv2.Canny(src_gray, 50, 150, apertureSize=3)
        ref_edges = cv2.Canny(ref_gray, 50, 150, apertureSize=3)
        
        print(f"  🔍 Step 2: Feature detection (ORB on edges)...")
        # Detect ORB features on edge maps (much better for line drawings!)
        orb = cv2.ORB_create(
            nfeatures=500,  # More features for better matching
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,  # Lower threshold to detect more
            patchSize=31
        )
        
        kp1, des1 = orb.detectAndCompute(src_edges, None)
        kp2, des2 = orb.detectAndCompute(ref_edges, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"  ⚠️  Not enough features (src={len(kp1) if kp1 else 0}, ref={len(kp2) if kp2 else 0})")
            print(f"  → Returning original (no registration)")
            return source.copy(), {
                'method': 'line_drawing',
                'success': False,
                'reason': 'Not enough edge features',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        print(f"  ✓ Found features: src={len(kp1)}, ref={len(kp2)}")
        
        print(f"  🔗 Step 3: Feature matching...")
        # Match features using BFMatcher with cross-check
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Ratio test
                    good_matches.append(m)
        
        print(f"  ✓ Good matches: {len(good_matches)}")
        
        if len(good_matches) < 10:
            print(f"  ⚠️  Not enough good matches ({len(good_matches)} < 10)")
            print(f"  → Returning original (no registration)")
            return source.copy(), {
                'method': 'line_drawing',
                'success': False,
                'reason': f'Not enough matches ({len(good_matches)})',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        print(f"  📊 Step 4: Estimate Similarity Transform (RANSAC)...")
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate Similarity Transform with RANSAC
        # This gives us X/Y translation + rotation + uniform scale (NO shear!)
        M, mask = cv2.estimateAffinePartial2D(
            src_pts, 
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,  # Stricter threshold for better accuracy
            maxIters=2000,
            confidence=0.99
        )
        
        if M is None:
            print(f"  ⚠️  RANSAC failed to estimate transform")
            print(f"  → Returning original (no registration)")
            return source.copy(), {
                'method': 'line_drawing',
                'success': False,
                'reason': 'RANSAC estimation failed',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        # Count inliers
        inliers = np.sum(mask) if mask is not None else 0
        print(f"  ✓ RANSAC inliers: {inliers}/{len(good_matches)}")
        
        print(f"  ✅ Step 5: Validate transformation...")
        # Extract and validate transformation parameters
        tx = M[0, 2]
        ty = M[1, 2]
        
        # Extract rotation and scale from the matrix
        # M = [s*cos(θ)  -s*sin(θ)  tx]
        #     [s*sin(θ)   s*cos(θ)  ty]
        a = M[0, 0]
        b = M[0, 1]
        
        scale = np.sqrt(a*a + b*b)
        rotation_rad = np.arctan2(b, a)
        rotation_deg = np.degrees(rotation_rad)
        
        print(f"    Translation: ({tx:.1f}, {ty:.1f})")
        print(f"    Rotation: {rotation_deg:.1f}°")
        print(f"    Scale: {scale:.2f}x")
        
        # Validation checks
        image_size = np.sqrt(source.shape[0]**2 + source.shape[1]**2)
        max_translation = image_size * 0.4  # Max 40% of diagonal
        
        rejection_reasons = []
        
        if abs(rotation_deg) > max_rotation_degrees:
            rejection_reasons.append(f'rotation too large ({rotation_deg:.1f}° > {max_rotation_degrees}°)')
        
        if scale < 0.6 or scale > 1.5:
            rejection_reasons.append(f'scale too extreme ({scale:.2f}x, allowed: 0.6-1.5x)')
        
        if abs(tx) > max_translation or abs(ty) > max_translation:
            rejection_reasons.append(f'translation too large ({tx:.1f},{ty:.1f} > {max_translation:.1f})')
        
        if inliers < 10:
            rejection_reasons.append(f'too few inliers ({inliers} < 10)')
        
        if rejection_reasons:
            print(f"  ❌ Registration rejected: {', '.join(rejection_reasons)}")
            print(f"  → Returning original (no registration)")
            return source.copy(), {
                'method': 'line_drawing',
                'success': False,
                'reason': 'Validation failed: ' + ', '.join(rejection_reasons),
                'rejected_params': {
                    'translation_x': float(tx),
                    'translation_y': float(ty),
                    'rotation_degrees': float(rotation_deg),
                    'scale': float(scale),
                    'inliers': int(inliers)
                },
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        print(f"  ✅ Validation passed! Applying transformation...")
        # Apply transformation
        registered = cv2.warpAffine(
            source,
            M,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
        )
        
        # Clean borders
        registered = self._clean_borders(registered, border_size=5)
        
        print(f"  🎉 Registration successful!")
        
        return registered, {
            'method': 'line_drawing',
            'success': True,
            'translation_x': float(tx),
            'translation_y': float(ty),
            'rotation_degrees': float(rotation_deg),
            'scale': float(scale),
            'inliers': int(inliers),
            'total_matches': len(good_matches)
        }
    
    def _register_ecc_OLD(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        motion_type: str = "euclidean",
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register using Enhanced Correlation Coefficient (ECC) with multi-angle initialization.
        
        ECC is a local optimizer, so we try multiple starting angles to find the best alignment.
        For simple line drawings, this dramatically improves registration accuracy.
        
        Args:
            source: Source image
            reference: Reference image
            motion_type: Type of motion ('euclidean', 'affine', 'translation')
            max_rotation_degrees: Maximum allowed rotation in degrees
            
        Returns:
            Tuple of (registered_image, info)
        """
        # Convert to grayscale if needed
        if len(source.shape) == 3:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source.copy()
            
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference.copy()
        
        # Ensure same size
        if source_gray.shape != ref_gray.shape:
            source_gray = cv2.resize(source_gray, 
                                    (ref_gray.shape[1], ref_gray.shape[0]))
        
        # NO contrast enhancement - keep it simple for line drawings!
        
        # Define motion model based on type
        if motion_type == "translation":
            # Only translation (X, Y shift)
            warp_mode = cv2.MOTION_TRANSLATION
        elif motion_type == "affine":
            # Full affine (translation + rotation + scale + shear)
            warp_mode = cv2.MOTION_AFFINE
        elif motion_type == "similarity":
            # Similarity: translation + rotation + UNIFORM scale (no shear/distortion)
            # Use feature-based method for similarity transform
            return self._register_similarity(source, reference, max_rotation_degrees)
        else:  # euclidean (default)
            # Euclidean (translation + rotation, NO scale)
            warp_mode = cv2.MOTION_EUCLIDEAN
        
        # For simple line drawings, use a simplified approach
        # Try only a few angles, not the full multi-angle search
        if warp_mode == cv2.MOTION_EUCLIDEAN:
            print(f"🔄 Simple registration for line drawings...")
            return self._register_simple_rotation(source, reference, source_gray, ref_gray, warp_mode, max_rotation_degrees)
        
        # Initialize the matrix
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Specify the number of iterations (for non-Euclidean motion types)
        number_of_iterations = 2000  # Balanced: not too slow, not too fast
        termination_eps = 1e-6  # Good enough convergence
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                   number_of_iterations, termination_eps)
        
        try:
            # Run ECC algorithm
            (cc, warp_matrix) = cv2.findTransformECC(
                ref_gray,
                source_gray,
                warp_matrix,
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=5
            )
            
            # Apply transformation with white background
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                registered = cv2.warpPerspective(
                    source,
                    warp_matrix,
                    (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
                )
            else:
                registered = cv2.warpAffine(
                    source,
                    warp_matrix,
                    (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
                )
            
            # Clean up borders to avoid edge artifacts
            registered = self._clean_borders(registered, border_size=5)
            
            # Extract transformation parameters
            info = self._extract_transformation_info(warp_matrix, warp_mode)
            info['correlation_coefficient'] = float(cc)
            info['method'] = 'ecc'
            info['success'] = True
            
            return registered, info
            
        except Exception as e:
            # If ECC fails, return original with error info
            return source.copy(), {
                'method': 'ecc',
                'success': False,
                'error': str(e),
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
    
    def _register_simple_rotation(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        source_gray: np.ndarray,
        ref_gray: np.ndarray,
        warp_mode: int,
        max_rotation_degrees: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simple rotation-only registration for line drawings.
        
        Just try a few discrete angles and pick the best one based on
        image similarity (correlation). No iterative optimization needed!
        
        Args:
            source: Source image (color)
            reference: Reference image (color)
            source_gray: Source image (grayscale)
            ref_gray: Reference image (grayscale)
            warp_mode: OpenCV motion type
            max_rotation_degrees: Maximum allowed rotation
            
        Returns:
            Best registration result
        """
        # Try only 3 angles: 0°, -15°, +15° (very simple!)
        angles_to_try = [0, -15, 15, -30, 30]
        
        best_score = -1
        best_warp = None
        best_angle = 0
        
        center = (source_gray.shape[1] / 2, source_gray.shape[0] / 2)
        
        print(f"  Testing {len(angles_to_try)} simple angles...")
        
        for angle_deg in angles_to_try:
            if abs(angle_deg) > max_rotation_degrees:
                continue
            
            try:
                # Simple rotation matrix
                angle_rad = np.radians(angle_deg)
                M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
                
                # Rotate source
                rotated = cv2.warpAffine(
                    source_gray,
                    M,
                    (source_gray.shape[1], source_gray.shape[0]),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )
                
                # Simple correlation score (faster than ECC!)
                # Use normalized cross-correlation
                score = cv2.matchTemplate(rotated, ref_gray, cv2.TM_CCORR_NORMED)[0][0]
                
                print(f"    {angle_deg:+4d}° → score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_warp = M.copy()
                    best_angle = angle_deg
                    
            except Exception as e:
                print(f"    {angle_deg:+4d}° → failed")
                continue
        
        if best_warp is None:
            # All failed, return original
            print(f"  ❌ All angles failed, no registration")
            return source.copy(), {
                'method': 'simple_rotation',
                'success': False,
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        print(f"  ✅ Best angle: {best_angle}° (score={best_score:.4f})")
        
        # Apply best rotation to color image
        registered = cv2.warpAffine(
            source,
            best_warp,
            (reference.shape[1], reference.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
        )
        
        # Clean borders
        registered = self._clean_borders(registered, border_size=5)
        
        return registered, {
            'method': 'simple_rotation',
            'success': True,
            'translation_x': float(best_warp[0, 2]),
            'translation_y': float(best_warp[1, 2]),
            'rotation_degrees': float(best_angle),
            'scale': 1.0,
            'score': float(best_score)
        }
    
    def _register_ecc_multi_angle(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        source_gray: np.ndarray,
        ref_gray: np.ndarray,
        warp_mode: int,
        max_rotation_degrees: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Try ECC with multiple initial rotation angles.
        
        This overcomes ECC's limitation as a local optimizer by trying multiple
        starting points and selecting the best result.
        
        Args:
            source: Source image (color)
            reference: Reference image (color)
            source_gray: Source image (grayscale, enhanced)
            ref_gray: Reference image (grayscale, enhanced)
            warp_mode: OpenCV motion type
            max_rotation_degrees: Maximum allowed rotation
            
        Returns:
            Best registration result
        """
        # Try fewer angles for speed: 0°, ±10°, ±20°, ±30° (7 instead of 13)
        angles_to_try = [0, 10, -10, 20, -20, 30, -30]
        
        best_cc = -1
        best_warp = None
        best_angle = 0
        
        center = (source_gray.shape[1] / 2, source_gray.shape[0] / 2)
        
        print(f"  Testing {len(angles_to_try)} angles: {angles_to_try}")
        
        for i, angle_deg in enumerate(angles_to_try):
            if abs(angle_deg) > max_rotation_degrees:
                continue
                
            try:
                # Create initial transformation matrix with this rotation
                angle_rad = np.radians(angle_deg)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                
                # Euclidean transformation matrix: [cos -sin tx]
                #                                  [sin  cos ty]
                warp_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0]
                ], dtype=np.float32)
                
                # Adjust translation to keep center fixed
                # This prevents the image from moving off-screen during rotation
                cx, cy = center
                warp_matrix[0, 2] = cx - cos_a * cx + sin_a * cy
                warp_matrix[1, 2] = cy - sin_a * cx - cos_a * cy
                
                # Run ECC from this starting point (reduced iterations for speed)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
                
                print(f"    [{i+1}/{len(angles_to_try)}] Trying {angle_deg:+3d}°...", end=" ")
                
                (cc, warp_matrix) = cv2.findTransformECC(
                    ref_gray,
                    source_gray,
                    warp_matrix,
                    warp_mode,
                    criteria,
                    inputMask=None,
                    gaussFiltSize=5
                )
                
                print(f"cc={cc:.4f}")
                
                # Keep track of best result
                if cc > best_cc:
                    best_cc = cc
                    best_warp = warp_matrix.copy()
                    best_angle = angle_deg
                    
            except Exception as e:
                # This angle didn't work, continue to next
                print(f"failed ({str(e)[:30]})")
                continue
        
        if best_warp is None:
            # All attempts failed, return original
            print(f"❌ All ECC angles failed, returning original")
            return source.copy(), {
                'method': 'ecc_multi_angle',
                'success': False,
                'error': 'All angles failed',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        print(f"✅ Best ECC result: angle={best_angle}° → cc={best_cc:.4f}")
        
        # Apply best transformation
        registered = cv2.warpAffine(
            source,
            best_warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
        )
        
        # Clean up borders
        registered = self._clean_borders(registered, border_size=5)
        
        # Extract transformation parameters
        info = self._extract_transformation_info(best_warp, warp_mode)
        info['correlation_coefficient'] = float(best_cc)
        info['method'] = 'ecc_multi_angle'
        info['initial_angle'] = best_angle
        info['success'] = True
        
        return registered, info
    
    def _register_similarity(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register using Similarity Transform.
        
        Similarity = Translation + Rotation + UNIFORM Scale (no shear/distortion)
        This is the best balance for hand-drawn images.
        
        INCLUDES VALIDATION: Rejects transformations that are too extreme (large scale changes, 
        excessive translation). Falls back to no registration if transformation is unrealistic.
        
        Args:
            source: Source image
            reference: Reference image
            max_rotation_degrees: Maximum allowed rotation in degrees
            
        Returns:
            Tuple of (registered_image, info)
        """
        # Convert to grayscale
        if len(source.shape) == 3:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source.copy()
            
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference.copy()
        
        # Ensure same size
        if source_gray.shape != ref_gray.shape:
            source_gray = cv2.resize(source_gray, 
                                    (ref_gray.shape[1], ref_gray.shape[0]))
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(source_gray, None)
        kp2, des2 = orb.detectAndCompute(ref_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Not enough features, try ECC instead
            print(f"⚠️  Not enough ORB features (kp1={len(kp1) if kp1 else 0}, kp2={len(kp2) if kp2 else 0}), trying ECC...")
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:min(50, len(matches))]
        
        if len(good_matches) < 4:
            # Not enough matches, try ECC instead
            print(f"⚠️  Not enough good matches ({len(good_matches)}), trying ECC...")
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate SIMILARITY transform (not full affine)
        # This constrains to: translation + rotation + uniform scale (no shear)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, ransacReprojThreshold=5.0)
        
        if M is None:
            # Could not estimate, try ECC instead
            print(f"⚠️  Could not estimate similarity transform, trying ECC...")
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Extract and validate transformation parameters
        info_temp = self._extract_transformation_info(M, cv2.MOTION_EUCLIDEAN)
        rotation_deg = abs(info_temp['rotation_degrees'])
        scale = info_temp['scale']
        tx = info_temp['translation_x']
        ty = info_temp['translation_y']
        
        # VALIDATION CHECKS: Reject only EXTREMELY unrealistic transformations
        image_diagonal = np.sqrt(source.shape[0]**2 + source.shape[1]**2)
        max_translation = image_diagonal * 0.5  # Max 50% of diagonal (relaxed from 30%)
        
        rejection_reasons = []
        
        if rotation_deg > max_rotation_degrees:
            rejection_reasons.append(f'rotation too large ({rotation_deg:.1f}° > {max_rotation_degrees}°)')
        
        # Relaxed scale limits: 0.5x - 2.0x (was 0.7x - 1.3x)
        if scale < 0.5 or scale > 2.0:
            rejection_reasons.append(f'scale too extreme ({scale:.2f}x, allowed: 0.5-2.0x)')
        
        if abs(tx) > max_translation or abs(ty) > max_translation:
            rejection_reasons.append(f'translation too large (tx={tx:.1f}, ty={ty:.1f}, max={max_translation:.1f})')
        
        if rejection_reasons:
            # Log rejection for debugging
            print(f"⚠️  Similarity rejected ({', '.join(rejection_reasons)}), trying ECC as fallback...")
            # Transformation is unrealistic - fallback to ECC with euclidean motion
            # ECC is more robust for simple line drawings
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Transformation is reasonable - apply it
        registered = cv2.warpAffine(
            source,
            M,
            (reference.shape[1], reference.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
        )
        
        # Clean up borders
        registered = self._clean_borders(registered, border_size=5)
        
        info = self._extract_transformation_info(M, cv2.MOTION_EUCLIDEAN)
        info['method'] = 'similarity'
        info['success'] = True
        info['num_matches'] = len(good_matches)
        info['validated'] = True
        
        return registered, info
    
    def _register_feature_based(
        self,
        source: np.ndarray,
        reference: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register using feature matching (ORB features).
        
        Good for images with distinct features.
        
        Args:
            source: Source image
            reference: Reference image
            
        Returns:
            Tuple of (registered_image, info)
        """
        # Convert to grayscale
        if len(source.shape) == 3:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source.copy()
            
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference.copy()
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(source_gray, None)
        kp2, des2 = orb.detectAndCompute(ref_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            # Not enough features, return original
            return source.copy(), {
                'method': 'feature',
                'success': False,
                'error': 'Not enough features detected',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:min(50, len(matches))]
        
        if len(good_matches) < 4:
            return source.copy(), {
                'method': 'feature',
                'success': False,
                'error': 'Not enough good matches',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find affine transformation
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if M is None:
            return source.copy(), {
                'method': 'feature',
                'success': False,
                'error': 'Could not estimate transformation',
                'translation_x': 0,
                'translation_y': 0,
                'rotation_degrees': 0,
                'scale': 1.0
            }
        
        # Apply transformation with white background
        registered = cv2.warpAffine(
            source,
            M,
            (reference.shape[1], reference.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(source.shape) == 3 else 255
        )
        
        # Clean up borders
        registered = self._clean_borders(registered, border_size=5)
        
        info = self._extract_transformation_info(M, cv2.MOTION_EUCLIDEAN)
        info['method'] = 'feature'
        info['success'] = True
        info['num_matches'] = len(good_matches)
        
        return registered, info
    
    def _register_simple(
        self,
        source: np.ndarray,
        reference: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simple registration: just resize to match reference.
        
        Args:
            source: Source image
            reference: Reference image
            
        Returns:
            Tuple of (registered_image, info)
        """
        if source.shape[:2] != reference.shape[:2]:
            registered = cv2.resize(source, 
                                  (reference.shape[1], reference.shape[0]))
        else:
            registered = source.copy()
        
        return registered, {
            'method': 'simple',
            'success': True,
            'translation_x': 0,
            'translation_y': 0,
            'rotation_degrees': 0,
            'scale': 1.0
        }
    
    def _thin_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Make lines 1-pixel thin and smooth using skeletonization.
        
        This is useful after scaling up, which makes lines thicker.
        Uses scikit-image's skeletonization for perfect 1px lines + Gaussian smoothing.
        
        Args:
            image: Image with potentially thick lines
            
        Returns:
            Image with 1-pixel thin, smooth lines
        """
        from skimage.morphology import skeletonize
        from skimage.filters import gaussian
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False
        
        # Step 1: Light Gaussian blur for smoothing (BEFORE skeletonization)
        # This removes small artifacts and smooths edges
        smoothed = gaussian(gray, sigma=0.5, preserve_range=True).astype(np.uint8)
        
        # Step 2: Threshold to binary (black lines on white)
        _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Step 3: Skeletonize - reduce to 1-pixel thin lines
        # Convert to boolean for skimage
        binary_bool = binary > 0
        skeleton_bool = skeletonize(binary_bool)
        
        # Convert back to uint8 (0 or 255)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        
        # Step 4: Invert back (white background, black lines)
        thinned = cv2.bitwise_not(skeleton)
        
        # Step 5: Light blur again for anti-aliasing (makes lines smoother)
        thinned = cv2.GaussianBlur(thinned, (3, 3), 0.5)
        
        # Convert back to color if needed
        if is_color:
            thinned_color = cv2.cvtColor(thinned, cv2.COLOR_GRAY2BGR)
            return thinned_color
        else:
            return thinned
    
    def _center_on_canvas(
        self,
        image: np.ndarray,
        target_size: tuple,
        value: int = 255
    ) -> np.ndarray:
        """
        Center an image on a canvas of target size.
        
        This prevents clipping when scaling up. Instead of resizing
        (which cuts off edges), we place the image centered on a
        white canvas.
        
        Args:
            image: Image to center
            target_size: (height, width) of target canvas
            value: Fill value for canvas (255=white, 0=black)
            
        Returns:
            Centered image on canvas
        """
        target_h, target_w = target_size
        src_h, src_w = image.shape[:2]
        
        # If image is larger than target, we need to crop/resize
        if src_h > target_h or src_w > target_w:
            # Scale down to fit
            scale = min(target_h / src_h, target_w / src_w)
            new_h = int(src_h * scale)
            new_w = int(src_w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            src_h, src_w = new_h, new_w
        
        # Create canvas
        if len(image.shape) == 3:
            canvas = np.full((target_h, target_w, image.shape[2]), value, dtype=image.dtype)
        else:
            canvas = np.full((target_h, target_w), value, dtype=image.dtype)
        
        # Calculate centered position
        y_offset = (target_h - src_h) // 2
        x_offset = (target_w - src_w) // 2
        
        # Place image on canvas
        canvas[y_offset:y_offset+src_h, x_offset:x_offset+src_w] = image
        
        return canvas
    
    def _clean_borders(
        self,
        image: np.ndarray,
        border_size: int = 5
    ) -> np.ndarray:
        """
        Clean up border artifacts by setting border pixels to white.
        
        Args:
            image: Image to clean
            border_size: Size of border to clean in pixels
            
        Returns:
            Cleaned image
        """
        cleaned = image.copy()
        
        # Set top and bottom borders to white
        cleaned[:border_size, :] = 255
        cleaned[-border_size:, :] = 255
        
        # Set left and right borders to white
        cleaned[:, :border_size] = 255
        cleaned[:, -border_size:] = 255
        
        return cleaned
    
    def _extract_transformation_info(
        self,
        matrix: np.ndarray,
        warp_mode: int
    ) -> Dict:
        """
        Extract human-readable transformation parameters.
        
        Args:
            matrix: Transformation matrix (2x3 or 3x3)
            warp_mode: OpenCV warp mode
            
        Returns:
            Dictionary with transformation parameters
        """
        if matrix.shape[0] == 2:
            # Affine/Euclidean transformation
            tx = float(matrix[0, 2])
            ty = float(matrix[1, 2])
            
            # Extract rotation and scale
            a = matrix[0, 0]
            b = matrix[0, 1]
            c = matrix[1, 0]
            d = matrix[1, 1]
            
            # For affine: scale can be different in x and y
            scale_x = np.sqrt(a*a + b*b)
            scale_y = np.sqrt(c*c + d*d)
            scale = (scale_x + scale_y) / 2  # Average scale
            
            rotation_rad = np.arctan2(b, a)
            rotation_deg = np.degrees(rotation_rad)
            
            return {
                'translation_x': tx,
                'translation_y': ty,
                'rotation_degrees': rotation_deg,
                'scale': scale,
                'scale_x': scale_x,
                'scale_y': scale_y
            }
        else:
            # Homography
            return {
                'translation_x': float(matrix[0, 2]),
                'translation_y': float(matrix[1, 2]),
                'rotation_degrees': 0,  # Complex to extract
                'scale': 1.0  # Complex to extract
            }

