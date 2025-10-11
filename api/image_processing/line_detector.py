"""
Line detection module using OpenCV.

This module implements line detection algorithms to extract line features
from hand-drawn images. It uses Hough Line Transform and edge detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import json
from .utils import preprocess_for_line_detection


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
        threshold: int = 100,
        min_line_length: int = 80,
        max_line_gap: int = 25
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
        
        # Merge similar lines to reduce duplicates
        merged_lines = self._merge_similar_lines(detected_lines)
        
        return merged_lines
    
    def _merge_similar_lines(self, lines: List[Tuple[int, int, int, int]], 
                            position_threshold: float = 15.0,
                            angle_threshold: float = 8.0) -> List[Tuple[int, int, int, int]]:
        """
        Merge lines that are very similar (likely duplicates from Hough Transform).
        
        Args:
            lines: List of lines
            position_threshold: Max distance to consider lines similar (pixels)
            angle_threshold: Max angle difference to consider lines similar (degrees)
            
        Returns:
            List of merged lines
        """
        if len(lines) == 0:
            return []
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
                
            # Find all similar lines
            similar = [line1]
            x1_1, y1_1, x2_1, y2_1 = line1
            angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1) * 180 / np.pi
            mid1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
            
            for j, line2 in enumerate(lines):
                if j <= i or j in used:
                    continue
                
                x1_2, y1_2, x2_2, y2_2 = line2
                angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2) * 180 / np.pi
                mid2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
                
                # Check if similar
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                    
                distance = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                
                if angle_diff < angle_threshold and distance < position_threshold:
                    similar.append(line2)
                    used.add(j)
            
            # Average all similar lines
            if len(similar) > 0:
                avg_x1 = int(np.mean([l[0] for l in similar]))
                avg_y1 = int(np.mean([l[1] for l in similar]))
                avg_x2 = int(np.mean([l[2] for l in similar]))
                avg_y2 = int(np.mean([l[3] for l in similar]))
                merged.append((avg_x1, avg_y1, avg_x2, avg_y2))
                used.add(i)
        
        return merged
    
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

