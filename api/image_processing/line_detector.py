"""
Line detection module using OpenCV.

This module implements line detection algorithms to extract line features
from hand-drawn images. It uses Hough Line Transform and edge detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import json


class LineDetector:
    """
    Line detector using Hough Line Transform.
    
    This class provides methods to detect lines in images and extract
    their features for comparison.
    """
    
    def __init__(
        self,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10
    ):
        """
        Initialize the line detector with parameters.
        
        Args:
            rho: Distance resolution in pixels
            theta: Angle resolution in radians
            threshold: Minimum number of votes (intersections)
            min_line_length: Minimum length of line
            max_line_gap: Maximum gap between line segments
        """
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def detect_lines(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect lines in a binary image using Probabilistic Hough Transform.
        
        Args:
            binary_image: Binary image (preprocessed)
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        lines = cv2.HoughLinesP(
            binary_image,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((int(x1), int(y1), int(x2), int(y2)))
        
        return detected_lines
    
    def detect_contours(self, binary_image: np.ndarray) -> List:
        """
        Detect contours in a binary image.
        
        Args:
            binary_image: Binary image (preprocessed)
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract line features from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing detected features
        """
        from image_processing.utils import preprocess_for_line_detection
        
        # Preprocess image
        binary = preprocess_for_line_detection(image)
        
        # Detect lines
        lines = self.detect_lines(binary)
        
        # Detect contours
        contours = self.detect_contours(binary)
        
        # Calculate statistics
        features = {
            'lines': lines,
            'num_lines': len(lines),
            'num_contours': len(contours),
            'image_shape': image.shape[:2],
            'line_lengths': [self._calculate_line_length(line) for line in lines],
            'line_angles': [self._calculate_line_angle(line) for line in lines]
        }
        
        return features
    
    def _calculate_line_length(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate Euclidean length of a line."""
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _calculate_line_angle(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate angle of a line in degrees."""
        x1, y1, x2, y2 = line
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    
    def features_to_json(self, features: Dict) -> str:
        """
        Convert features dictionary to JSON string.
        
        Args:
            features: Feature dictionary
            
        Returns:
            JSON string representation
        """
        # Convert numpy types to native Python types for JSON serialization
        serializable_features = {
            'lines': features['lines'],
            'num_lines': int(features['num_lines']),
            'num_contours': int(features['num_contours']),
            'image_shape': [int(x) for x in features['image_shape']],
            'line_lengths': [float(x) for x in features['line_lengths']],
            'line_angles': [float(x) for x in features['line_angles']]
        }
        return json.dumps(serializable_features)
    
    def features_from_json(self, json_str: str) -> Dict:
        """
        Load features from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            Features dictionary
        """
        return json.loads(json_str)

