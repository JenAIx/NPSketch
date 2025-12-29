"""
Target Normalization for Regression Tasks

Handles normalization/denormalization of target values for better training.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class TargetNormalizer:
    """
    Normalizes target values to improve training stability and performance.
    
    Supports different normalization strategies:
    - min_max: Scale to [0, 1] range
    - standardize: Z-score normalization (mean=0, std=1)
    - none: No normalization (raw values)
    """
    
    def __init__(
        self,
        method: str = 'min_max',
        value_range: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize normalizer.
        
        Args:
            method: 'min_max', 'standardize', or 'none'
            value_range: (min, max) tuple for min_max method.
                        If None, computed from data.
        """
        self.method = method
        self.value_range = value_range
        
        # Statistics (computed from data or provided)
        self.min_value = None
        self.max_value = None
        self.mean = None
        self.std = None
        
        if value_range is not None:
            self.min_value, self.max_value = value_range
    
    def fit(self, targets: np.ndarray):
        """
        Compute normalization statistics from training data.
        
        Args:
            targets: Array of target values
        """
        if self.method == 'min_max':
            if self.min_value is None:
                self.min_value = float(np.min(targets))
            if self.max_value is None:
                self.max_value = float(np.max(targets))
            
            logger.info(f"Min-Max normalization: [{self.min_value}, {self.max_value}] → [0, 1]")
            
        elif self.method == 'standardize':
            self.mean = float(np.mean(targets))
            self.std = float(np.std(targets))
            
            logger.info(f"Standardization: mean={self.mean:.2f}, std={self.std:.2f}")
            
        elif self.method == 'none':
            logger.info("No normalization applied")
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def transform(self, targets: np.ndarray) -> np.ndarray:
        """
        Normalize target values.
        
        Args:
            targets: Raw target values
        
        Returns:
            Normalized target values
        """
        if self.method == 'min_max':
            if self.min_value is None or self.max_value is None:
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            # Scale to [0, 1]
            normalized = (targets - self.min_value) / (self.max_value - self.min_value)
            return normalized
            
        elif self.method == 'standardize':
            if self.mean is None or self.std is None:
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            # Z-score normalization
            normalized = (targets - self.mean) / (self.std + 1e-8)
            return normalized
            
        elif self.method == 'none':
            return targets
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            normalized: Normalized predictions
        
        Returns:
            Original-scale predictions
        """
        if self.method == 'min_max':
            if self.min_value is None or self.max_value is None:
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            # Scale back from [0, 1]
            original = normalized * (self.max_value - self.min_value) + self.min_value
            return original
            
        elif self.method == 'standardize':
            if self.mean is None or self.std is None:
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            # Reverse z-score
            original = normalized * (self.std + 1e-8) + self.mean
            return original
            
        elif self.method == 'none':
            return normalized
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def get_config(self) -> Dict:
        """Get normalizer configuration for saving."""
        return {
            'method': self.method,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'mean': self.mean,
            'std': self.std,
            'value_range': self.value_range
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TargetNormalizer':
        """Create normalizer from saved configuration."""
        normalizer = cls(
            method=config['method'],
            value_range=config.get('value_range')
        )
        normalizer.min_value = config.get('min_value')
        normalizer.max_value = config.get('max_value')
        normalizer.mean = config.get('mean')
        normalizer.std = config.get('std')
        return normalizer


# Preset normalizers for common score ranges
def get_normalizer_for_feature(feature_name: str) -> TargetNormalizer:
    """
    Get appropriate normalizer for a specific feature.
    
    Args:
        feature_name: Name of the target feature
    
    Returns:
        Configured TargetNormalizer
    """
    # Known score ranges for neuropsychological tests
    score_ranges = {
        'MMSE': (0, 30),
        'ACE': (0, 100),
        'Total_Score': (0, 60),  # Your current case
        'Park_Total': (0, 60),
        'MoCA': (0, 30),
        'CDR': (0, 3),
    }
    
    if feature_name in score_ranges:
        min_val, max_val = score_ranges[feature_name]
        logger.info(f"Using predefined range for {feature_name}: [{min_val}, {max_val}]")
        return TargetNormalizer(method='min_max', value_range=(min_val, max_val))
    else:
        logger.info(f"No predefined range for {feature_name}, will compute from data")
        return TargetNormalizer(method='min_max')

