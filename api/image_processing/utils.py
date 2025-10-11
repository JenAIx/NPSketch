"""
Utility functions for image processing.

This module provides helper functions for image loading, conversion,
normalization, and other common image processing tasks.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        Image as numpy array in BGR format
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def normalize_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Normalize image to standard size while maintaining aspect ratio.
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height)
        
    Returns:
        Normalized image with padding if necessary
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with padding
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    
    # Center the image on canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    """
    Convert numpy image to bytes.
    
    Args:
        image: Image as numpy array
        format: Output format (PNG, JPEG, etc.)
        
    Returns:
        Image as bytes
    """
    is_success, buffer = cv2.imencode(f'.{format.lower()}', image)
    if not is_success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


def preprocess_for_line_detection(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for line detection.
    
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Apply adaptive thresholding
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Preprocessed binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return binary


def create_visualization(
    original: np.ndarray,
    detected_lines: list,
    reference_lines: Optional[list] = None,
    matches: Optional[list] = None
) -> np.ndarray:
    """
    Create visualization overlay showing detected lines and matches.
    
    Args:
        original: Original image
        detected_lines: List of detected lines
        reference_lines: Optional reference lines
        matches: Optional list of matching line pairs
        
    Returns:
        Visualization image with overlays
    """
    vis = original.copy()
    
    # Draw detected lines in blue
    for line in detected_lines:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw reference lines in green if provided
    if reference_lines is not None:
        for line in reference_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw matches in yellow if provided
    if matches is not None:
        for match in matches:
            x1, y1, x2, y2 = match
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    
    return vis

