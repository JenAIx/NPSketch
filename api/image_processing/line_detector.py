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
        threshold: int = 20,           # Optimized for 256x256 images
        min_line_length: int = 80,     # Optimized: filters noise, keeps main lines
        max_line_gap: int = 30         # Optimized: moderate merging
    ):
        """
        Initialize the line detector with parameters.
        
        Args:
            rho: Distance resolution in pixels
            theta: Angle resolution in radians
            threshold: Minimum number of votes (intersections, lowered for sensitivity: 60)
            min_line_length: Minimum length of line (lowered to 60)
            max_line_gap: Maximum gap between line segments (increased to 50 to connect broken lines!)
        
        Note:
            These parameters were re-optimized to prevent long lines from being split into segments.
            The key change is max_line_gap: 30 → 50, which helps connect broken line segments.
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
                            position_threshold: float = 50.0,  # VERY aggressive for fragmented lines
                            angle_threshold: float = 2.0) -> List[Tuple[int, int, int, int]]:  # Strict for precise matching
        """
        Merge lines that are very similar (likely duplicates from Hough Transform).
        
        Uses a multi-pass strategy:
        1. First pass: Merge collinear segments (same angle, on same line)
        2. Second pass: Merge nearby parallel lines
        
        Args:
            lines: List of lines
            position_threshold: Max distance to consider lines similar (pixels)
            angle_threshold: Max angle difference to consider lines similar (degrees)
            
        Returns:
            List of merged lines
        """
        if len(lines) == 0:
            return []
        
        # FIRST PASS: Aggressive collinear merging
        # This handles the 27 fragmented 45° segments
        collinear_merged = self._merge_collinear_segments(lines)
        
        # SECOND PASS: Standard similarity merging
        merged = []
        used = set()
        
        for i, line1 in enumerate(collinear_merged):
            if i in used:
                continue
                
            # Find all similar lines
            similar = [line1]
            x1_1, y1_1, x2_1, y2_1 = line1
            angle1 = np.arctan2(y2_1 - y1_1, x2_1 - x1_1) * 180 / np.pi
            mid1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
            
            for j, line2 in enumerate(collinear_merged):
                if j <= i or j in used:
                    continue
                
                x1_2, y1_2, x2_2, y2_2 = line2
                angle2 = np.arctan2(y2_2 - y1_2, x2_2 - x1_2) * 180 / np.pi
                mid2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
                
                # Check if similar
                # Normalize angles to 0-90° range (lines have no direction)
                norm_angle1 = abs(angle1) if abs(angle1) <= 90 else 180 - abs(angle1)
                norm_angle2 = abs(angle2) if abs(angle2) <= 90 else 180 - abs(angle2)
                angle_diff = abs(norm_angle1 - norm_angle2)
                    
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
    
    def _merge_collinear_segments(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Aggressively merge collinear line segments that lie on the same infinite line.
        
        This is specifically designed to handle fragmented lines (like the 27 segments
        of the 45° X-crossing) that have nearly identical angles but are broken into
        many small pieces.
        
        Args:
            lines: List of line segments
            
        Returns:
            List of merged lines
        """
        if len(lines) == 0:
            return []
        
        # Group lines by angle (within 1.5 degrees)
        angle_groups = []
        angle_threshold = 1.5
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Normalize to 0-180 range
            if angle < 0:
                angle += 180
            
            # Find or create angle group
            found_group = False
            for group in angle_groups:
                group_angle = group['angle']
                if abs(angle - group_angle) < angle_threshold:
                    group['lines'].append(line)
                    found_group = True
                    break
            
            if not found_group:
                angle_groups.append({'angle': angle, 'lines': [line]})
        
        # For each angle group, merge collinear segments
        merged_lines = []
        for group in angle_groups:
            group_lines = group['lines']
            
            if len(group_lines) == 1:
                merged_lines.append(group_lines[0])
                continue
            
            # For each unique infinite line within this angle group
            # merge all segments that lie on it
            while len(group_lines) > 0:
                # Start with first line
                seed_line = group_lines.pop(0)
                x1_s, y1_s, x2_s, y2_s = seed_line
                
                # Collect all collinear segments
                collinear = [seed_line]
                remaining = []
                
                for line in group_lines:
                    x1, y1, x2, y2 = line
                    
                    # First check: Do the lines intersect? If yes, they're crossing, not collinear
                    if self._lines_intersect(seed_line, line):
                        remaining.append(line)
                        continue
                    
                    # Check if line is collinear with seed (point-to-line distance)
                    # Use all 4 endpoints
                    d1 = self._point_to_line_distance((x1, y1), seed_line)
                    d2 = self._point_to_line_distance((x2, y2), seed_line)
                    
                    # STRICT collinearity check: BOTH endpoints must be very close
                    # This prevents merging crossing lines (like the X) that have similar angles
                    if d1 < 5 and d2 < 5:  # VERY strict: only 5px tolerance
                        collinear.append(line)
                    else:
                        remaining.append(line)
                
                group_lines = remaining
                
                # Merge all collinear segments into one long line
                # Find the two points that are furthest apart
                all_points = []
                for line in collinear:
                    all_points.append((line[0], line[1]))
                    all_points.append((line[2], line[3]))
                
                max_dist = -1
                best_pair = None
                for i in range(len(all_points)):
                    for j in range(i + 1, len(all_points)):
                        dist = np.sqrt((all_points[i][0] - all_points[j][0])**2 + 
                                     (all_points[i][1] - all_points[j][1])**2)
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (all_points[i], all_points[j])
                
                if best_pair:
                    merged_lines.append((best_pair[0][0], best_pair[0][1], 
                                       best_pair[1][0], best_pair[1][1]))
        
        return merged_lines
    
    def _point_to_line_distance(self, point: Tuple[int, int], line: Tuple[int, int, int, int]) -> float:
        """
        Calculate perpendicular distance from a point to a line segment.
        
        Args:
            point: (x, y) coordinates
            line: (x1, y1, x2, y2) line segment
            
        Returns:
            Distance in pixels
        """
        px, py = point
        x1, y1, x2, y2 = line
        
        # Line length
        line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_len == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Perpendicular distance
        distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_len
        return distance
    
    def _lines_intersect(self, line1: Tuple[int, int, int, int], line2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two line segments intersect (cross each other).
        
        Uses the cross product method to detect intersection.
        
        Args:
            line1: First line (x1, y1, x2, y2)
            line2: Second line (x1, y1, x2, y2)
            
        Returns:
            True if lines intersect, False otherwise
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate direction vectors
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)
        
        # Lines intersect if endpoints are on opposite sides
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
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
        Extract line features from an image with categorization.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing detected features with line categorization
        """
        # Preprocess image
        binary = preprocess_for_line_detection(image)
        
        # Detect lines
        lines = self.detect_lines(binary)
        
        # Categorize lines
        categorized = self.categorize_lines(lines)
        
        # Detect contours
        contours = self.detect_contours(binary)
        
        # Calculate statistics
        features = {
            'lines': lines,
            'num_lines': len(lines),
            'num_contours': len(contours),
            'image_shape': image.shape[:2],
            'line_lengths': [self._calculate_line_length(line) for line in lines],
            'line_angles': [self._calculate_line_angle(line) for line in lines],
            # Add categorization
            'categorized_lines': categorized,
            'line_counts': categorized['counts']
        }
        
        return features
    
    def _calculate_line_length(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate Euclidean length of a line."""
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def validate_against_description(self, features: Dict, description_path: str = '/app/templates/image_description.json') -> Dict:
        """
        Validate detected lines against expected description.
        
        Args:
            features: Extracted features with categorization
            description_path: Path to image description JSON
            
        Returns:
            Validation results with matches and mismatches
        """
        import json
        import os
        
        if not os.path.exists(description_path):
            return {'error': 'Description file not found', 'validated': False}
        
        with open(description_path, 'r') as f:
            description = json.load(f)
        
        expected = description['expected_lines']
        detected = features['line_counts']
        
        validation = {
            'validated': True,
            'perfect_match': True,
            'expected': expected,
            'detected': detected,
            'differences': {},
            'summary': []
        }
        
        # Check each category
        for category in ['horizontal', 'vertical', 'diagonal']:
            exp = expected.get(category, 0)
            det = detected.get(category, 0)
            diff = det - exp
            
            if diff != 0:
                validation['perfect_match'] = False
                validation['differences'][category] = {
                    'expected': exp,
                    'detected': det,
                    'difference': diff
                }
                
                if diff > 0:
                    validation['summary'].append(f'{category}: +{diff} extra lines')
                else:
                    validation['summary'].append(f'{category}: {diff} missing lines')
            else:
                validation['summary'].append(f'{category}: ✓ correct ({det})')
        
        # Check total
        total_exp = description.get('total_lines', sum(expected.values()))
        total_det = detected.get('total', 0)
        validation['total_match'] = (total_exp == total_det)
        
        return validation
    
    def categorize_lines(self, lines: List[Tuple[int, int, int, int]]) -> Dict[str, List]:
        """
        Categorize lines into horizontal, vertical, and diagonal.
        
        Lines have no inherent direction, so angles are normalized to 0-90° range.
        For example, 119.9° is normalized to 60.1° (180 - 119.9).
        
        Args:
            lines: List of lines as (x1, y1, x2, y2)
            
        Returns:
            Dictionary with categorized lines:
            {
                'horizontal': [...],
                'vertical': [...],
                'diagonal': [...]
            }
        """
        horizontal = []
        vertical = []
        diagonal = []
        
        for line in lines:
            angle = self._calculate_line_angle(line)
            
            # Normalize angle to 0-90° range (lines have no direction)
            norm_angle = abs(angle)
            if norm_angle > 90:
                norm_angle = 180 - norm_angle
            
            # Categorize based on normalized angle
            # Horizontal: angle close to 0° (0-15°) - wider range for slight variations
            if norm_angle < 15:
                horizontal.append(line)
            # Vertical: angle close to 90° (75-90°) - wider range for slight variations
            elif norm_angle > 75:
                vertical.append(line)
            # Diagonal: everything else (15-75°, typically 55-70° for roof)
            else:
                diagonal.append(line)
        
        return {
            'horizontal': horizontal,
            'vertical': vertical,
            'diagonal': diagonal,
            'counts': {
                'horizontal': len(horizontal),
                'vertical': len(vertical),
                'diagonal': len(diagonal),
                'total': len(lines)
            }
        }
    
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
            'line_angles': [float(x) for x in features['line_angles']],
            # Add categorization
            'line_counts': features.get('line_counts', {}),
            'categorized_lines': {
                k: v for k, v in features.get('categorized_lines', {}).items() 
                if k != 'counts'  # Skip counts as it's in line_counts
            }
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

