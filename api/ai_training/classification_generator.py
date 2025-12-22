"""
Classification Generator for Training Data

Generates balanced class labels for features with non-overlapping ranges.
"""

import numpy as np
from typing import List, Dict, Tuple


def generate_balanced_classes(
    scores: np.ndarray,
    num_classes: int,
    method: str = "quantile"
) -> Dict:
    """
    Generate balanced class assignments for scores.
    
    Args:
        scores: Array of score values
        num_classes: Number of classes to create (2-10)
        method: "quantile" for equal-sample or "equal-width" for equal ranges
    
    Returns:
        Dictionary with class data and boundaries
    """
    if len(scores) == 0:
        raise ValueError("No scores provided")
    
    if num_classes < 2 or num_classes > 10:
        raise ValueError("num_classes must be between 2 and 10")
    
    total_samples = len(scores)
    
    if method == "equal-width":
        return _generate_equal_width_classes(scores, num_classes, total_samples)
    else:  # quantile (default)
        return _generate_equal_sample_classes(scores, num_classes, total_samples)


def _generate_equal_sample_classes(
    scores: np.ndarray,
    num_classes: int,
    total_samples: int
) -> Dict:
    """
    Generate classes with approximately equal number of samples.
    Keeps identical scores together (no splitting).
    """
    # Step 1: Get unique scores and their counts
    unique_scores, score_counts = np.unique(scores, return_counts=True)
    
    # Step 2: Greedy assignment - balance while keeping same scores together
    target_per_class = total_samples / num_classes
    tolerance = 0.15  # Allow 15% deviation from target
    
    class_assignments = []  # List of (score_indices, count)
    current_class_indices = []
    current_class_count = 0
    
    for idx, (score, count) in enumerate(zip(unique_scores, score_counts)):
        count = int(count)
        classes_created = len(class_assignments)
        classes_remaining = num_classes - classes_created
        
        # If this is the last class needed, add all remaining
        if classes_remaining == 1:
            current_class_indices.append(idx)
            current_class_count += count
            # Add all remaining scores
            for j in range(idx + 1, len(unique_scores)):
                current_class_indices.append(j)
                current_class_count += int(score_counts[j])
            class_assignments.append((current_class_indices, current_class_count))
            break
        
        # Check if we should close current class
        will_be_count = current_class_count + count
        close_class = False
        
        if current_class_count > 0:
            # Close if we've exceeded target
            if will_be_count > target_per_class * (1 + tolerance):
                close_class = True
            # Close if we're at/above target and adding more would be worse
            elif current_class_count >= target_per_class * (1 - tolerance):
                # Check if keeping or not keeping is closer to target
                diff_if_add = abs(will_be_count - target_per_class)
                diff_if_close = abs(current_class_count - target_per_class)
                if diff_if_close <= diff_if_add:
                    close_class = True
        
        if close_class and classes_remaining > 1:
            # Close current class
            class_assignments.append((current_class_indices, current_class_count))
            # Start new class
            current_class_indices = [idx]
            current_class_count = count
        else:
            # Add to current class
            current_class_indices.append(idx)
            current_class_count += count
    
    # Close last class if not already closed
    if current_class_indices and len(class_assignments) < num_classes:
        class_assignments.append((current_class_indices, current_class_count))
    
    # Step 3: Build class_data with non-overlapping ranges
    class_data = []
    for class_id, (score_indices, count) in enumerate(class_assignments):
        class_scores = unique_scores[score_indices]
        class_min = int(round(float(class_scores[0])))
        class_max = int(round(float(class_scores[-1])))
        
        # First class should start at 0 (or known minimum)
        if class_id == 0:
            class_min = 0
        
        class_data.append({
            "id": int(class_id),
            "min": class_min,
            "max": class_max,
            "count": int(count),
            "percentage": round(float(count / total_samples) * 100.0, 2),
            "label": f"Class_{class_id}"
        })
    
    # Build boundaries
    boundaries = [int(class_data[0]["min"])]
    for cls in class_data:
        boundaries.append(int(cls["max"]))
    
    return {
        "method": "quantile",
        "boundaries": boundaries,
        "classes": class_data,
        "actual_num_classes": len(class_data),
        "requested_num_classes": num_classes
    }


def _generate_equal_width_classes(
    scores: np.ndarray,
    num_classes: int,
    total_samples: int
) -> Dict:
    """
    Generate classes with equal-width ranges.
    """
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    
    # Create equal-width boundaries
    boundaries = np.linspace(min_score, max_score, num_classes + 1)
    
    class_data = []
    for class_id in range(num_classes):
        class_min = boundaries[class_id]
        class_max = boundaries[class_id + 1]
        
        # Count samples in this range
        if class_id == num_classes - 1:
            mask = (scores >= class_min) & (scores <= class_max)
        else:
            mask = (scores >= class_min) & (scores < class_max)
        
        count = int(np.sum(mask))
        percentage = (count / total_samples) * 100.0
        
        class_data.append({
            "id": int(class_id),
            "min": int(round(class_min)),
            "max": int(round(class_max)),
            "count": count,
            "percentage": round(percentage, 2),
            "label": f"Class_{class_id}"
        })
    
    return {
        "method": "equal-width",
        "boundaries": [float(b) for b in boundaries],
        "classes": class_data,
        "actual_num_classes": num_classes,
        "requested_num_classes": num_classes
    }

