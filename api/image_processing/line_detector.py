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
        threshold: int = 18,           # Balanced sensitivity
        min_line_length: int = 35,     # Balanced to avoid noise
        max_line_gap: int = 35,        # Moderate gap closing
        final_min_length: int = 30     # Final filter: no lines below 30px
    ):
        """
        Initialize the line detector with parameters.
        
        Args:
            rho: Distance resolution in pixels
            theta: Angle resolution in radians
            threshold: Minimum number of votes (lowered to 15 for better sensitivity)
            min_line_length: Initial minimum length (25px allows detection of segments)
            max_line_gap: Maximum gap between segments (40px to better connect broken lines)
            final_min_length: Final minimum length filter (30px to remove short artifacts)
        
        Note:
            Two-stage filtering: First detect with min_line_length=25, then merge,
            then filter out anything below final_min_length=30px.
        """
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.final_min_length = final_min_length
    
    def detect_lines(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect lines using ITERATIVE DETECTION with PIXEL SUBTRACTION.
        
        Strategy:
        1. Detect lines (longest first)
        2. Remove pixels of detected line from image (with buffer)
        3. Repeat until no more lines found
        4. Filter overlaps and duplicates
        
        This ensures:
        - Longest lines are detected first
        - No duplicate/overlapping detections
        - Better handling of crossing lines
        
        Args:
            binary_image: Binary image (preprocessed)
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        return self._detect_lines_iterative(binary_image)
    
    def _detect_lines_iterative(self, binary_image: np.ndarray, max_iterations: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Iteratively detect lines with MULTI-PASS strategy:
        
        Pass 1 (Iterations 1-10): Detect strong/long lines (strict)
        Pass 2 (Iterations 11-20): Detect weak/short lines (relaxed)
        
        After each line:
        - Dilate line mask (expand by 2-3px)
        - Subtract from image
        - Search for next line
        
        Args:
            binary_image: Binary image
            max_iterations: Maximum number of iterations
            
        Returns:
            List of detected lines
        """
        # Work on a copy
        working_image = binary_image.copy()
        all_lines = []
        
        print(f"  🔄 Iterative line detection (max {max_iterations} iterations)...")
        print(f"  📊 Image has {np.sum(working_image > 0)} black pixels")
        
        for iteration in range(max_iterations):
            # Adaptive threshold: Start strict, then relax
            if iteration < 10:
                # Pass 1: Strong lines (strict)
                current_threshold = max(12, self.threshold - 3)
                current_min_length = self.min_line_length
            else:
                # Pass 2: Weak lines (relaxed)
                current_threshold = max(8, self.threshold - 8)
                current_min_length = max(25, self.min_line_length - 10)
            
            # Detect lines
            lines = cv2.HoughLinesP(
                working_image,
                rho=self.rho,
                theta=self.theta,
                threshold=current_threshold,
                minLineLength=current_min_length,
                maxLineGap=self.max_line_gap
            )
            
            if lines is None or len(lines) == 0:
                # Check remaining pixels
                remaining = np.sum(working_image > 0)
                print(f"    Iteration {iteration + 1}: No more lines found ({remaining} pixels remain). Stopping.")
                break
            
            # Convert and sort by length (longest first)
            detected = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                detected.append((int(x1), int(y1), int(x2), int(y2), length))
            
            detected.sort(key=lambda x: x[4], reverse=True)
            
            # Take the longest line from this iteration
            if len(detected) > 0:
                longest = detected[0]
                x1, y1, x2, y2, length = longest
                line = (x1, y1, x2, y2)
                
                # Check if this line overlaps significantly with existing lines
                if not self._overlaps_with_existing(line, all_lines):
                    all_lines.append(line)
                    print(f"    Iteration {iteration + 1} [Pass {1 if iteration < 10 else 2}]: Found line (length={length:.1f}px) → Total: {len(all_lines)}")
                    
                    # Remove pixels with dilated mask
                    self._erase_line_pixels(working_image, line, buffer=4)
                else:
                    print(f"    Iteration {iteration + 1}: Line overlaps, skipping.")
                    # Still erase to avoid infinite loop
                    self._erase_line_pixels(working_image, line, buffer=4)
            
            # Stop if we have enough lines
            if len(all_lines) >= 12:
                print(f"    Reached {len(all_lines)} lines, stopping.")
                break
        
        print(f"  ✅ Detected {len(all_lines)} lines after {min(iteration + 1, max_iterations)} iterations")
        
        # Final filter: Remove lines shorter than final_min_length
        filtered_lines = []
        for line in all_lines:
            length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
            if length >= self.final_min_length:
                filtered_lines.append(line)
        
        print(f"  ✅ After length filter (>={self.final_min_length}px): {len(filtered_lines)} lines")
        
        return filtered_lines
    
    def _erase_line_pixels(self, image: np.ndarray, line: Tuple[int, int, int, int], buffer: int = 5):
        """
        Erase pixels along a line from the image (with dilation + subtraction).
        
        Strategy:
        1. Draw the line on a temporary mask
        2. Dilate the line (expand by 2-3px) to include nearby pixels
        3. Subtract from image to remove the entire line region
        
        This ensures:
        - Complete removal of line pixels
        - No artifacts/remnants left behind
        - Crossing points are handled better
        
        Args:
            image: Binary image (modified in-place)
            line: Line as (x1, y1, x2, y2)
            buffer: Buffer size around line to erase (pixels)
        """
        x1, y1, x2, y2 = line
        
        # Create a temporary mask for this line
        mask = np.zeros_like(image)
        
        # Draw the line on the mask (thicker than before)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=buffer * 2)
        
        # DILATE: Expand the line by 2-3 pixels to capture nearby artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # SUBTRACT: Remove the dilated line region from the image
        # Where mask is white (255), set image to black (0)
        image[dilated_mask > 0] = 0
    
    def _overlaps_with_existing(
        self, 
        line: Tuple[int, int, int, int], 
        existing_lines: List[Tuple[int, int, int, int]],
        angle_threshold: float = 8.0,
        position_threshold: float = 25.0
    ) -> bool:
        """
        Check if a line significantly overlaps with existing lines.
        
        Special handling for crossing lines (X pattern):
        - Crossing lines have different angles (e.g., 45° vs 135°)
        - They share a crossing point but are NOT duplicates
        
        Args:
            line: New line to check
            existing_lines: List of already detected lines
            angle_threshold: Max angle difference in degrees (8° for crossing tolerance)
            position_threshold: Max distance in pixels (25px tighter)
            
        Returns:
            True if line overlaps significantly
        """
        if len(existing_lines) == 0:
            return False
        
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Normalize angle to 0-180° (keep direction to distinguish crossing lines)
        norm_angle = angle % 180
        
        for existing in existing_lines:
            ex1, ey1, ex2, ey2 = existing
            e_angle = np.arctan2(ey2 - ey1, ex2 - ex1) * 180 / np.pi
            e_mid = ((ex1 + ex2) / 2, (ey1 + ey2) / 2)
            
            # Normalize existing angle to 0-180°
            e_norm_angle = e_angle % 180
            
            # Check angle difference
            angle_diff = abs(norm_angle - e_norm_angle)
            
            # Special case: Crossing lines (e.g., 45° and 135°)
            # These have ~90° difference and should NOT be considered overlapping
            if 80 <= angle_diff <= 100:
                # This is a crossing line, not a duplicate!
                continue
            
            # Check distance
            distance = np.sqrt((mid[0] - e_mid[0])**2 + (mid[1] - e_mid[1])**2)
            
            # If similar angle AND close position → overlap
            if angle_diff < angle_threshold and distance < position_threshold:
                return True
        
        return False
    
    def _merge_similar_lines(self, lines: List[Tuple[int, int, int, int]], 
                            position_threshold: float = 40.0,  # More conservative
                            angle_threshold: float = 3.0) -> List[Tuple[int, int, int, int]]:  # Slightly more tolerant
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

