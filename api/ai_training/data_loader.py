"""
Training Data Loader

Loads training data from database for CNN training.
"""

import numpy as np
from PIL import Image
import io
import json
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from database import TrainingDataImage


class TrainingDataLoader:
    """Load and prepare training data for CNN."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_available_features(self) -> Dict:
        """Get all unique feature keys across all images."""
        images = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        all_features = set()
        feature_stats = {}
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                for key in features.keys():
                    all_features.add(key)
                    if key not in feature_stats:
                        feature_stats[key] = {'count': 0, 'min': float('inf'), 'max': float('-inf'), 'values': []}
                    
                    value = float(features[key])
                    feature_stats[key]['count'] += 1
                    feature_stats[key]['min'] = min(feature_stats[key]['min'], value)
                    feature_stats[key]['max'] = max(feature_stats[key]['max'], value)
                    feature_stats[key]['values'].append(value)
            except:
                pass
        
        # Calculate means
        for key in feature_stats:
            values = feature_stats[key]['values']
            feature_stats[key]['mean'] = np.mean(values) if values else 0
            feature_stats[key]['std'] = np.std(values) if values else 0
            del feature_stats[key]['values']  # Remove raw values
        
        return {
            'features': sorted(list(all_features)),
            'stats': feature_stats
        }
    
    def get_dataset_info(self) -> Dict:
        """Get information about available training data."""
        total = self.db.query(TrainingDataImage).count()
        with_features = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None),
            TrainingDataImage.features_data != '{}',
            TrainingDataImage.features_data != 'null'
        ).count()
        
        # Count by format
        mat_count = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.source_format == 'MAT'
        ).count()
        
        ocs_count = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.source_format == 'OCS'
        ).count()
        
        # Count by task
        copy_count = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.task_type == 'COPY'
        ).count()
        
        recall_count = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.task_type == 'RECALL'
        ).count()
        
        return {
            'total_images': total,
            'labeled_images': with_features,
            'unlabeled_images': total - with_features,
            'by_format': {
                'MAT': mat_count,
                'OCS': ocs_count
            },
            'by_task': {
                'COPY': copy_count,
                'RECALL': recall_count
            }
        }
    
    def load_training_data(
        self, 
        target_feature: str,
        train_split: float = 0.8,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Load training data from database.
        
        Args:
            target_feature: Feature to predict (e.g., 'Total_Score')
            train_split: Percentage for training (0.8 = 80%)
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, image_ids)
        """
        # Load images with the target feature
        images = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        X = []
        y = []
        image_ids = []
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                if target_feature in features:
                    # Load processed image
                    pil_img = Image.open(io.BytesIO(img.processed_image_data))
                    img_array = np.array(pil_img)
                    
                    # Convert to grayscale if RGB
                    if len(img_array.shape) == 3:
                        img_array = np.mean(img_array, axis=2)
                    
                    # Normalize to 0-1
                    img_array = img_array.astype(np.float32) / 255.0
                    
                    X.append(img_array)
                    y.append(float(features[target_feature]))
                    image_ids.append(img.id)
            except Exception as e:
                print(f"Error loading image {img.id}: {e}")
        
        if len(X) == 0:
            raise ValueError(f"No images found with feature '{target_feature}'")
        
        X = np.array(X)
        y = np.array(y)
        
        # Add channel dimension for CNN (height, width, channels)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        
        # Train/test split
        np.random.seed(random_seed)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * train_split)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test, image_ids

