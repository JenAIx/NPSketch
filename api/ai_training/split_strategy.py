"""
Train/Test Split Strategy with Stratification for Regression

Ensures balanced distribution of target values across train/test sets.
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy import stats
from utils.logger import get_logger

logger = get_logger(__name__)


def analyze_distribution(y_values: np.ndarray) -> Dict:
    """
    Analyze distribution of target values.
    
    Args:
        y_values: Array of target values
    
    Returns:
        Distribution statistics
    """
    return {
        'count': len(y_values),
        'mean': float(np.mean(y_values)),
        'std': float(np.std(y_values)),
        'min': float(np.min(y_values)),
        'max': float(np.max(y_values)),
        'median': float(np.median(y_values)),
        'q25': float(np.percentile(y_values, 25)),
        'q75': float(np.percentile(y_values, 75)),
        'range': float(np.max(y_values) - np.min(y_values))
    }


def create_bins(y_values: np.ndarray, n_bins: int = 4, method: str = 'quantile') -> np.ndarray:
    """
    Create bins for stratification.
    
    Args:
        y_values: Target values
        n_bins: Number of bins
        method: 'quantile' (equal frequency) or 'equal' (equal width)
    
    Returns:
        Bin assignments for each sample
    """
    if method == 'quantile':
        # Equal frequency bins (quantiles)
        # Ensures each bin has roughly equal number of samples
        bins = np.percentile(y_values, np.linspace(0, 100, n_bins + 1))
    elif method == 'equal':
        # Equal width bins
        bins = np.linspace(y_values.min(), y_values.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning method: {method}")
    
    # Assign each value to a bin
    # Use digitize: returns bin index for each value
    bin_assignments = np.digitize(y_values, bins[1:-1])  # Exclude first and last edge
    
    return bin_assignments


def stratified_split_regression(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    n_bins: int = 4,
    random_seed: int = 42,
    min_samples_per_bin: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create stratified train/test split for regression.
    
    Uses binning to ensure balanced distribution of target values.
    
    Args:
        X: Features array (images)
        y: Target values array
        train_split: Fraction for training
        n_bins: Number of bins for stratification
        random_seed: Random seed for reproducibility
        min_samples_per_bin: Minimum samples required per bin
    
    Returns:
        (X_train, X_test, y_train, y_test, split_info)
    """
    np.random.seed(random_seed)
    
    # Analyze distribution
    dist_info = analyze_distribution(y)
    
    # Create bins
    bin_assignments = create_bins(y, n_bins=n_bins, method='quantile')
    
    # Count samples per bin
    unique_bins, bin_counts = np.unique(bin_assignments, return_counts=True)
    
    # Check if enough samples per bin
    small_bins = []
    for bin_idx, count in zip(unique_bins, bin_counts):
        if count < min_samples_per_bin:
            small_bins.append((bin_idx, count))
    
    # If bins are too small, use random split instead
    if small_bins:
        logger.warning(f"{len(small_bins)} bins have < {min_samples_per_bin} samples")
        logger.warning("Falling back to random split")
        return random_split(X, y, train_split, random_seed)
    
    # Stratified split: For each bin, split proportionally
    train_indices = []
    test_indices = []
    
    for bin_idx in unique_bins:
        # Get indices for this bin
        bin_mask = bin_assignments == bin_idx
        bin_indices = np.where(bin_mask)[0]
        
        # Shuffle indices within bin
        np.random.shuffle(bin_indices)
        
        # Split this bin
        split_point = int(len(bin_indices) * train_split)
        
        train_indices.extend(bin_indices[:split_point])
        test_indices.extend(bin_indices[split_point:])
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Create splits
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Analyze split quality
    split_info = validate_split(y_train, y_test, y)
    split_info['method'] = 'stratified'
    split_info['n_bins'] = n_bins
    split_info['bin_counts'] = {int(k): int(v) for k, v in zip(unique_bins, bin_counts)}
    
    return X_train, X_test, y_train, y_test, split_info


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Simple random split (fallback).
    
    Args:
        X: Features
        y: Targets
        train_split: Train fraction
        random_seed: Random seed
    
    Returns:
        (X_train, X_test, y_train, y_test, split_info)
    """
    np.random.seed(random_seed)
    
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * train_split)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    split_info = validate_split(y_train, y_test, y)
    split_info['method'] = 'random'
    
    return X_train, X_test, y_train, y_test, split_info


def validate_split(y_train: np.ndarray, y_test: np.ndarray, y_all: np.ndarray = None) -> Dict:
    """
    Validate train/test split quality.
    
    Checks if distributions are similar.
    
    Args:
        y_train: Training targets
        y_test: Test targets
        y_all: All targets (optional, for overall comparison)
    
    Returns:
        Validation statistics with warnings
    """
    train_dist = analyze_distribution(y_train)
    test_dist = analyze_distribution(y_test)
    
    warnings = []
    
    # Check mean difference
    mean_diff = abs(train_dist['mean'] - test_dist['mean'])
    mean_diff_pct = (mean_diff / train_dist['mean'] * 100) if train_dist['mean'] != 0 else 0
    
    if mean_diff_pct > 20:
        warnings.append(f"Mean difference: {mean_diff_pct:.1f}% (should be < 20%)")
    
    # Check std difference
    std_diff = abs(train_dist['std'] - test_dist['std'])
    std_diff_pct = (std_diff / train_dist['std'] * 100) if train_dist['std'] != 0 else 0
    
    if std_diff_pct > 30:
        warnings.append(f"Std deviation difference: {std_diff_pct:.1f}% (should be < 30%)")
    
    # Check range coverage
    if test_dist['min'] < train_dist['min'] or test_dist['max'] > train_dist['max']:
        warnings.append("Test set has values outside training range (extrapolation required)")
    
    # Kolmogorov-Smirnov test (distribution similarity)
    if len(y_train) > 2 and len(y_test) > 2:
        ks_statistic, ks_pvalue = stats.ks_2samp(y_train, y_test)
        
        if ks_pvalue < 0.05:
            warnings.append(f"Distributions significantly different (KS p-value: {ks_pvalue:.3f})")
    
    return {
        'train_distribution': train_dist,
        'test_distribution': test_dist,
        'mean_difference': float(mean_diff),
        'mean_difference_pct': float(mean_diff_pct),
        'std_difference': float(std_diff),
        'std_difference_pct': float(std_diff_pct),
        'warnings': warnings,
        'is_balanced': len(warnings) == 0
    }


def stratified_split_classification(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    random_seed: int = 42,
    min_samples_per_class: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Create stratified train/test split for classification.
    
    Ensures each class is proportionally represented in both train and test sets.
    
    Args:
        X: Features array (images)
        y: Class labels array (integers)
        train_split: Fraction for training
        random_seed: Random seed for reproducibility
        min_samples_per_class: Minimum samples required per class
    
    Returns:
        (X_train, X_test, y_train, y_test, split_info)
    """
    np.random.seed(random_seed)
    
    # Get unique classes and counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    num_classes = len(unique_classes)
    
    # Check if enough samples per class
    small_classes = []
    for cls, count in zip(unique_classes, class_counts):
        if count < min_samples_per_class:
            small_classes.append((int(cls), int(count)))
    
    # If classes are too small, use random split
    if small_classes:
        logger.warning(f"{len(small_classes)} classes have < {min_samples_per_class} samples")
        logger.warning(f"Classes: {small_classes}")
        logger.warning("Falling back to random split")
        return random_split(X, y, train_split, random_seed)
    
    # Stratified split: For each class, split proportionally
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        # Get indices for this class
        class_mask = y == cls
        class_indices = np.where(class_mask)[0]
        
        # Shuffle indices within class
        np.random.shuffle(class_indices)
        
        # Split this class
        split_point = int(len(class_indices) * train_split)
        
        # Ensure at least 1 sample in test if possible
        if split_point == len(class_indices) and len(class_indices) > 1:
            split_point = len(class_indices) - 1
        
        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Create splits
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Analyze class distribution in splits
    train_class_counts = {int(cls): int(np.sum(y_train == cls)) for cls in unique_classes}
    test_class_counts = {int(cls): int(np.sum(y_test == cls)) for cls in unique_classes}
    
    warnings = []
    
    # Check if each class is represented in both splits
    for cls in unique_classes:
        if train_class_counts[int(cls)] == 0:
            warnings.append(f"Class {int(cls)} has no samples in training set")
        if test_class_counts[int(cls)] == 0:
            warnings.append(f"Class {int(cls)} has no samples in test set")
    
    # Check class balance (should be similar proportions in train and test)
    for cls in unique_classes:
        train_pct = train_class_counts[int(cls)] / len(y_train) * 100
        test_pct = test_class_counts[int(cls)] / len(y_test) * 100
        diff = abs(train_pct - test_pct)
        
        if diff > 15:  # More than 15% difference
            warnings.append(f"Class {int(cls)}: {diff:.1f}% imbalance between train and test")
    
    split_info = {
        'method': 'stratified_classification',
        'num_classes': int(num_classes),
        'train_distribution': {
            'count': len(y_train),
            'class_counts': train_class_counts
        },
        'test_distribution': {
            'count': len(y_test),
            'class_counts': test_class_counts
        },
        'warnings': warnings,
        'is_balanced': len(warnings) == 0
    }
    
    return X_train, X_test, y_train, y_test, split_info


def stratified_group_split(
    y: np.ndarray,
    groups: List[str],
    train_split: float = 0.8,
    n_bins: int = 4,
    random_seed: int = 42,
    is_classification: bool = False,
    force_train_prefixes: Tuple[str, ...] = ('SYNTH_',),
    group_label: str = 'patient_id'
) -> Tuple[List[int], List[int], Dict]:
    """
    Group-aware stratified train/val split (e.g. patient-level).

    All samples sharing the same group key (typically patient_id) are assigned
    to the SAME set, preventing leakage between train and val (e.g. FC0/FC1
    drawings of the same patient, or COPY/RECALL of the same patient).

    Stratification works on group representatives:
    - regression: median target value per group, binned into quantile bins
    - classification: majority class per group

    Within each bin, groups are assigned greedily by cumulative sample count so
    the train fraction holds on the sample level (groups vary in size).

    Args:
        y: Target values, one per sample
        groups: Group key per sample (same length as y). Samples with a key
                starting with one of force_train_prefixes (e.g. synthetic
                images) are always assigned to train.
        train_split: Fraction of samples for training
        n_bins: Number of quantile bins (regression only)
        random_seed: Random seed for reproducibility
        is_classification: Stratify by class labels instead of value bins
        force_train_prefixes: Group-key prefixes that are forced into train
        group_label: Name of the group key (for logging/split_info only)

    Returns:
        (train_indices, val_indices, split_info)
    """
    if len(y) != len(groups):
        raise ValueError(f"y ({len(y)}) and groups ({len(groups)}) must have the same length")

    np.random.seed(random_seed)

    # Build group -> sample indices mapping (insertion order)
    group_to_indices: Dict[str, List[int]] = {}
    for idx, key in enumerate(groups):
        group_to_indices.setdefault(str(key), []).append(idx)

    # Separate groups that are forced into train (e.g. synthetic images)
    forced_train_keys = [k for k in group_to_indices
                         if any(k.startswith(p) for p in force_train_prefixes)]
    splittable_keys = [k for k in group_to_indices if k not in set(forced_train_keys)]

    if len(splittable_keys) < 2:
        raise ValueError(f"Need at least 2 groups to split, got {len(splittable_keys)}")

    # Representative target value per group
    def representative(key: str) -> float:
        values = [y[i] for i in group_to_indices[key]]
        if is_classification:
            # Majority class
            uniq, counts = np.unique(values, return_counts=True)
            return float(uniq[np.argmax(counts)])
        return float(np.median(values))

    reps = np.array([representative(k) for k in splittable_keys])

    # Assign each group to a stratification bin
    if is_classification:
        bin_assignments = reps  # class label IS the bin
    else:
        if float(reps.max() - reps.min()) == 0:
            bin_assignments = np.zeros(len(reps))
            logger.warning("All group representatives identical - using single bin (grouped random split)")
        else:
            bin_assignments = create_bins(reps, n_bins=n_bins, method='quantile').astype(float)

    # Split groups bin-wise, greedily by cumulative sample count
    train_keys: List[str] = []
    val_keys: List[str] = []

    for bin_value in np.unique(bin_assignments):
        bin_keys = [splittable_keys[i] for i in np.where(bin_assignments == bin_value)[0]]
        np.random.shuffle(bin_keys)

        bin_total = sum(len(group_to_indices[k]) for k in bin_keys)
        target_train = train_split * bin_total

        bin_train: List[str] = []
        bin_val: List[str] = []
        running = 0
        for k in bin_keys:
            if running < target_train:
                bin_train.append(k)
                running += len(group_to_indices[k])
            else:
                bin_val.append(k)

        # Ensure both sides are non-empty when the bin has >= 2 groups
        if len(bin_keys) >= 2:
            if not bin_val:
                bin_val.append(bin_train.pop())
            elif not bin_train:
                bin_train.append(bin_val.pop())

        train_keys.extend(bin_train)
        val_keys.extend(bin_val)

    train_keys.extend(forced_train_keys)

    # Hard leakage check: no group may appear on both sides
    overlap = set(train_keys) & set(val_keys)
    if overlap:
        raise RuntimeError(f"Group split produced overlapping groups (leakage!): {sorted(overlap)[:5]}...")

    train_indices = [i for k in train_keys for i in group_to_indices[k]]
    val_indices = [i for k in val_keys for i in group_to_indices[k]]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    if not train_indices or not val_indices:
        raise ValueError(f"Group split produced an empty set "
                         f"(train={len(train_indices)}, val={len(val_indices)})")

    y_train = np.asarray([y[i] for i in train_indices])
    y_val = np.asarray([y[i] for i in val_indices])

    if is_classification:
        unique_classes = np.unique(y)
        train_class_counts = {int(c): int(np.sum(y_train == c)) for c in unique_classes}
        val_class_counts = {int(c): int(np.sum(y_val == c)) for c in unique_classes}

        warnings = []
        for c in unique_classes:
            if train_class_counts[int(c)] == 0:
                warnings.append(f"Class {int(c)} has no samples in training set")
            if val_class_counts[int(c)] == 0:
                warnings.append(f"Class {int(c)} has no samples in validation set")

        split_info = {
            'method': 'stratified_group_classification',
            'num_classes': int(len(unique_classes)),
            'train_distribution': {'count': len(y_train), 'class_counts': train_class_counts},
            'test_distribution': {'count': len(y_val), 'class_counts': val_class_counts},
            'warnings': warnings,
            'is_balanced': len(warnings) == 0
        }
    else:
        split_info = validate_split(y_train, y_val, y)
        split_info['method'] = 'stratified_group'
        split_info['n_bins'] = n_bins

    split_info['group_key'] = group_label
    split_info['n_groups_total'] = len(group_to_indices)
    split_info['n_groups_train'] = len(set(train_keys))
    split_info['n_groups_val'] = len(set(val_keys))
    split_info['n_groups_forced_train'] = len(forced_train_keys)
    split_info['group_overlap'] = 0

    logger.info(f"Group split ({group_label}): "
                f"{split_info['n_groups_train']} groups / {len(train_indices)} samples -> train, "
                f"{split_info['n_groups_val']} groups / {len(val_indices)} samples -> val, "
                f"group overlap: 0 (verified)")

    return train_indices, val_indices, split_info


def get_split_recommendation(n_samples: int, target_range: float) -> Dict:
    """
    Get recommended split strategy based on dataset size.
    
    Args:
        n_samples: Number of samples
        target_range: Range of target values (max - min)
    
    Returns:
        Recommendation dict
    """
    if n_samples < 20:
        return {
            'strategy': 'random',
            'reason': 'Too few samples for stratification',
            'n_bins': 2,
            'min_samples_needed': 20
        }
    elif n_samples < 50:
        return {
            'strategy': 'stratified',
            'reason': 'Small dataset benefits from stratification',
            'n_bins': 2,
            'min_samples_needed': 50
        }
    elif n_samples < 100:
        return {
            'strategy': 'stratified',
            'reason': 'Good dataset size',
            'n_bins': 3,
            'min_samples_needed': 100
        }
    else:
        return {
            'strategy': 'stratified',
            'reason': 'Large dataset',
            'n_bins': 4,
            'min_samples_needed': None
        }

