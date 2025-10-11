"""
Image registration module for aligning drawings with references.

This module implements image registration techniques to align uploaded images
with reference images before line detection, improving matching accuracy.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class ImageRegistration:
    """
    Handles image registration for aligning drawings with references.
    
    Uses feature-based and intensity-based registration methods to find
    the optimal transformation (translation, rotation, scale) to align images.
    """
    
    def __init__(self):
        """Initialize the image registration."""
        pass
    
    def register_images(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        method: str = "ecc",
        motion_type: str = "euclidean",
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register source image to reference image.
        
        Args:
            source: Source image to be aligned (uploaded drawing)
            reference: Reference image to align to
            method: Registration method ('ecc', 'feature', 'simple')
            motion_type: Type of motion model ('euclidean', 'affine', 'translation')
            max_rotation_degrees: Maximum allowed rotation in degrees
            
        Returns:
            Tuple of (registered_image, registration_info)
        """
        if method == "ecc":
            return self._register_ecc(source, reference, motion_type, max_rotation_degrees)
        elif method == "feature":
            return self._register_feature_based(source, reference)
        else:
            return self._register_simple(source, reference)
    
    def _register_ecc(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        motion_type: str = "euclidean",
        max_rotation_degrees: float = 30.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register using Enhanced Correlation Coefficient (ECC).
        
        This method is good for aligning similar images with translation,
        rotation, and scaling differences.
        
        Args:
            source: Source image
            reference: Reference image
            motion_type: Type of motion ('euclidean', 'affine', 'translation')
            
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
        
        # Initialize the matrix
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Specify the number of iterations
        number_of_iterations = 1000
        termination_eps = 1e-6
        
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
            # Not enough features, fallback to ECC euclidean
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches
        good_matches = matches[:min(50, len(matches))]
        
        if len(good_matches) < 4:
            # Fallback to ECC
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate SIMILARITY transform (not full affine)
        # This constrains to: translation + rotation + uniform scale (no shear)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        
        if M is None:
            # Fallback to ECC
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
        # Check rotation constraint
        info_temp = self._extract_transformation_info(M, cv2.MOTION_EUCLIDEAN)
        rotation_deg = abs(info_temp['rotation_degrees'])
        
        if rotation_deg > max_rotation_degrees:
            # Rotation too large, reject and use euclidean instead
            return self._register_ecc(source, reference, "euclidean", max_rotation_degrees)
        
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
        info['method'] = 'similarity'
        info['success'] = True
        info['num_matches'] = len(good_matches)
        
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

