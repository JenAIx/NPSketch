"""
Image processing library for NPSketch.

This package provides line detection, comparison, and visualization
functionality for comparing hand-drawn images to reference templates.
"""

from .line_detector import LineDetector
from .comparator import LineComparator
from .image_registration import ImageRegistration
from .utils import (
    load_image_from_bytes,
    normalize_image,
    image_to_bytes,
    preprocess_for_line_detection,
    create_visualization
)

__all__ = [
    'LineDetector',
    'LineComparator',
    'ImageRegistration',
    'load_image_from_bytes',
    'normalize_image',
    'image_to_bytes',
    'preprocess_for_line_detection',
    'create_visualization'
]


