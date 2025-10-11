"""
Line comparison module for evaluating drawings.

This module implements algorithms to compare detected lines from an uploaded
image with reference lines, calculating similarity metrics and identifying
correct, missing, and extra lines.
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment


class LineComparator:
    """
    Compares detected lines with reference lines.
    
    This class implements various similarity metrics and matching algorithms
    to evaluate how well an uploaded drawing matches a reference.
    """
    
    def __init__(
        self,
        position_tolerance: float = 120.0,  # Re-optimized: was 100.0
        angle_tolerance: float = 50.0,      # Re-optimized: was 45.0
        length_tolerance: float = 0.8,      # Re-optimized: was 0.7
        similarity_threshold: float = 0.5   # Lowered from 0.7 for better matching
    ):
        """
        Initialize comparator with tolerance thresholds.
        
        Args:
            position_tolerance: Maximum distance in pixels for position match (optimized: 120px)
            angle_tolerance: Maximum angle difference in degrees (optimized: 50°)
            length_tolerance: Maximum relative length difference (0-1, optimized: 0.8)
            similarity_threshold: Minimum similarity score for a match (lowered to 0.5)
        
        Note:
            Now uses Hungarian Algorithm for optimal line matching instead of greedy approach.
            This ensures the best global assignment of detected lines to reference lines.
        """
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.length_tolerance = length_tolerance
        self.similarity_threshold = similarity_threshold
    
    def compare_lines(
        self,
        detected_lines: List[Tuple[int, int, int, int]],
        reference_lines: List[Tuple[int, int, int, int]]
    ) -> Dict:
        """
        Compare detected lines with reference lines using Hungarian Algorithm.
        
        Uses optimal bipartite matching to find the best global assignment of
        detected lines to reference lines, rather than greedy matching.
        
        Args:
            detected_lines: List of detected lines as (x1, y1, x2, y2)
            reference_lines: List of reference lines as (x1, y1, x2, y2)
            
        Returns:
            Dictionary containing comparison results
        """
        if len(detected_lines) == 0 or len(reference_lines) == 0:
            # Handle empty cases
            return {
                'correct_lines': 0,
                'missing_lines': len(reference_lines),
                'extra_lines': len(detected_lines),
                'similarity_score': 0.0,
                'matches': [],
                'matched_detected_indices': [],
                'matched_reference_indices': []
            }
        
        # Build similarity matrix (cost matrix)
        # Higher similarity = lower cost, so use (1 - similarity)
        n_detected = len(detected_lines)
        n_reference = len(reference_lines)
        cost_matrix = np.ones((n_detected, n_reference))
        
        for i, det_line in enumerate(detected_lines):
            for j, ref_line in enumerate(reference_lines):
                similarity = self._calculate_line_similarity(det_line, ref_line)
                cost_matrix[i, j] = 1.0 - similarity  # Convert to cost
        
        # Use Hungarian Algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter matches by similarity threshold
        matched_detected = set()
        matched_reference = set()
        matches = []
        
        for i, j in zip(row_ind, col_ind):
            similarity = 1.0 - cost_matrix[i, j]
            if similarity >= self.similarity_threshold:
                matched_detected.add(i)
                matched_reference.add(j)
                matches.append((i, j, similarity))
        
        # Calculate metrics
        correct_lines = len(matches)
        missing_lines = len(reference_lines) - len(matched_reference)
        extra_lines = len(detected_lines) - len(matched_detected)
        
        # Calculate overall similarity score
        if len(reference_lines) > 0:
            similarity_score = correct_lines / len(reference_lines)
        else:
            similarity_score = 0.0
        
        return {
            'correct_lines': correct_lines,
            'missing_lines': missing_lines,
            'extra_lines': extra_lines,
            'similarity_score': similarity_score,
            'matches': matches,
            'matched_detected_indices': list(matched_detected),
            'matched_reference_indices': list(matched_reference)
        }
    
    def _calculate_line_similarity(
        self,
        line1: Tuple[int, int, int, int],
        line2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate similarity between two lines.
        
        Returns a score between 0 and 1, where 1 means identical lines.
        
        Args:
            line1: First line as (x1, y1, x2, y2)
            line2: Second line as (x1, y1, x2, y2)
            
        Returns:
            Similarity score (0-1)
        """
        # Calculate position similarity (distance between midpoints)
        mid1 = ((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2)
        mid2 = ((line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2)
        distance = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
        position_sim = max(0, 1 - distance / self.position_tolerance)
        
        # Calculate angle similarity
        angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
        angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])
        angle_diff = abs(angle1 - angle2) * 180 / np.pi
        # Handle angles wrapping around
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        angle_sim = max(0, 1 - angle_diff / self.angle_tolerance)
        
        # Calculate length similarity
        length1 = np.sqrt((line1[2] - line1[0])**2 + (line1[3] - line1[1])**2)
        length2 = np.sqrt((line2[2] - line2[0])**2 + (line2[3] - line2[1])**2)
        if length1 > 0 and length2 > 0:
            length_ratio = min(length1, length2) / max(length1, length2)
            length_sim = max(0, (length_ratio - (1 - self.length_tolerance)) / self.length_tolerance)
        else:
            length_sim = 0
        
        # Weighted average of similarities
        # Position is most important (50%), angle secondary (30%), length tertiary (20%)
        similarity = 0.5 * position_sim + 0.3 * angle_sim + 0.2 * length_sim
        
        return similarity
    
    def calculate_overall_metrics(
        self,
        detected_features: Dict,
        reference_features: Dict
    ) -> Dict:
        """
        Calculate overall comparison metrics.
        
        Args:
            detected_features: Features from detected image
            reference_features: Features from reference image
            
        Returns:
            Dictionary with comparison metrics
        """
        detected_lines = detected_features.get('lines', [])
        reference_lines = reference_features.get('lines', [])
        
        comparison = self.compare_lines(detected_lines, reference_lines)
        
        return {
            'correct_lines': comparison['correct_lines'],
            'missing_lines': comparison['missing_lines'],
            'extra_lines': comparison['extra_lines'],
            'similarity_score': comparison['similarity_score'],
            'total_reference_lines': len(reference_lines),
            'total_detected_lines': len(detected_lines)
        }


