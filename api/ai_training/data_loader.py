"""
Training Data Loader

Loads training data from database for CNN training.
Supports data augmentation for improved model generalization.
"""

import numpy as np
from PIL import Image
import io
import json
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from database import TrainingDataImage
from pathlib import Path

try:
    from .split_strategy import stratified_split_regression, validate_split, get_split_recommendation
except ImportError:
    # Fallback for direct execution
    try:
        from ai_training.split_strategy import stratified_split_regression, validate_split, get_split_recommendation
    except ImportError:
        # Last resort fallback - simple random split with complete return values
        def stratified_split_regression(X, y, train_split=0.8, n_bins=4, random_seed=42, min_samples_per_bin=2):
            np.random.seed(random_seed)
            indices = np.random.permutation(len(X))
            split_idx = int(len(X) * train_split)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            # Create distributions (matching real function signature)
            split_info = {
                'warnings': ['Using fallback random split (split_strategy module not available)'],
                'method': 'random',
                'train_distribution': {
                    'mean': float(np.mean(y_train)) if len(y_train) > 0 else 0,
                    'std': float(np.std(y_train)) if len(y_train) > 0 else 0,
                    'min': float(np.min(y_train)) if len(y_train) > 0 else 0,
                    'max': float(np.max(y_train)) if len(y_train) > 0 else 0
                },
                'test_distribution': {
                    'mean': float(np.mean(y_test)) if len(y_test) > 0 else 0,
                    'std': float(np.std(y_test)) if len(y_test) > 0 else 0,
                    'min': float(np.min(y_test)) if len(y_test) > 0 else 0,
                    'max': float(np.max(y_test)) if len(y_test) > 0 else 0
                }
            }
            
            return X[train_indices], X[test_indices], y_train, y_test, split_info
        
        def validate_split(y_train, y_test, y_all=None):
            # Minimal fallback that won't break code
            return {
                'warnings': [],
                'method': 'random',
                'train_distribution': {
                    'mean': float(np.mean(y_train)) if len(y_train) > 0 else 0,
                    'std': float(np.std(y_train)) if len(y_train) > 0 else 0,
                    'min': float(np.min(y_train)) if len(y_train) > 0 else 0,
                    'max': float(np.max(y_train)) if len(y_train) > 0 else 0
                },
                'test_distribution': {
                    'mean': float(np.mean(y_test)) if len(y_test) > 0 else 0,
                    'std': float(np.std(y_test)) if len(y_test) > 0 else 0,
                    'min': float(np.min(y_test)) if len(y_test) > 0 else 0,
                    'max': float(np.max(y_test)) if len(y_test) > 0 else 0
                }
            }
        
        def get_split_recommendation(n_samples, target_range):
            return {'strategy': 'random', 'reason': 'Fallback', 'n_bins': 2}


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
        custom_class_info = {}  # Track Custom_Class classifications
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                for key in features.keys():
                    # Skip Custom_Class - handle separately
                    if key == "Custom_Class":
                        # Parse Custom_Class structure
                        custom_classes = features["Custom_Class"]
                        for num_classes, class_data in custom_classes.items():
                            feature_key = f"Custom_Class_{num_classes}"
                            all_features.add(feature_key)
                            
                            if feature_key not in custom_class_info:
                                custom_class_info[feature_key] = {
                                    'num_classes': num_classes,
                                    'count': 0,
                                    'names': set()
                                }
                            
                            custom_class_info[feature_key]['count'] += 1
                            if class_data.get('name_custom'):
                                custom_class_info[feature_key]['names'].add(class_data['name_custom'])
                        continue
                    
                    # Handle regular numeric features
                    all_features.add(key)
                    if key not in feature_stats:
                        feature_stats[key] = {'count': 0, 'min': float('inf'), 'max': float('-inf'), 'values': []}
                    
                    try:
                        value = float(features[key])
                        feature_stats[key]['count'] += 1
                        feature_stats[key]['min'] = min(feature_stats[key]['min'], value)
                        feature_stats[key]['max'] = max(feature_stats[key]['max'], value)
                        feature_stats[key]['values'].append(value)
                    except (ValueError, TypeError):
                        # Not a numeric feature, skip stats
                        pass
            except:
                pass
        
        # Calculate means and median for numeric features
        for key in feature_stats:
            values = feature_stats[key]['values']
            feature_stats[key]['mean'] = float(np.mean(values)) if values else 0
            feature_stats[key]['median'] = float(np.median(values)) if values else 0
            feature_stats[key]['std'] = float(np.std(values)) if values else 0
            del feature_stats[key]['values']  # Remove raw values
        
        # Add stats for Custom_Class features
        for key, info in custom_class_info.items():
            feature_stats[key] = {
                'count': info['count'],
                'num_classes': info['num_classes'],
                'class_names': ', '.join(sorted(info['names'])) if info['names'] else 'N/A',
                'min': 0,
                'max': int(info['num_classes']) - 1,
                'mean': 0,
                'median': 0,
                'std': 0
            }
        
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
        
        # Count by format - dynamically get all unique formats
        from sqlalchemy import func
        format_counts = self.db.query(
            TrainingDataImage.source_format,
            func.count(TrainingDataImage.id)
        ).group_by(TrainingDataImage.source_format).all()
        
        by_format = {fmt: count for fmt, count in format_counts if fmt}
        
        # Count by task
        task_counts = self.db.query(
            TrainingDataImage.task_type,
            func.count(TrainingDataImage.id)
        ).group_by(TrainingDataImage.task_type).all()
        
        by_task = {task: count for task, count in task_counts if task}
        
        return {
            'total_images': total,
            'labeled_images': with_features,
            'unlabeled_images': total - with_features,
            'by_format': by_format,
            'by_task': by_task
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
        
        # Check if classification
        is_classification_mode = target_feature.startswith('Custom_Class_')
        if is_classification_mode:
            num_classes_str = target_feature.replace('Custom_Class_', '')
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                
                # Check if feature exists
                has_feature = False
                target_value = None
                
                if is_classification_mode:
                    if "Custom_Class" in features and num_classes_str in features.get("Custom_Class", {}):
                        has_feature = True
                        target_value = float(features["Custom_Class"][num_classes_str]["label"])
                else:
                    if target_feature in features:
                        has_feature = True
                        target_value = float(features[target_feature])
                
                if has_feature and target_value is not None:
                    # Load processed image
                    pil_img = Image.open(io.BytesIO(img.processed_image_data))
                    img_array = np.array(pil_img)
                    
                    # Convert to grayscale if RGB
                    if len(img_array.shape) == 3:
                        img_array = np.mean(img_array, axis=2)
                    
                    # Normalize to 0-1
                    img_array = img_array.astype(np.float32) / 255.0
                    
                    X.append(img_array)
                    y.append(target_value)
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
        
        # Choose split strategy based on task type
        if is_classification_mode:
            # For classification: stratify by actual class labels
            print(f"\nSplit Strategy: stratified_classification")
            print(f"Reason: Classification task - stratify by class labels")
            print(f"Number of classes: {len(np.unique(y))}")
            
            from ai_training.split_strategy import stratified_split_classification
            X_train, X_test, y_train, y_test, split_info = stratified_split_classification(
                X, y,
                train_split=train_split,
                random_seed=random_seed
            )
        else:
            # For regression: stratify by binning continuous values
            recommendation = get_split_recommendation(len(X), y.max() - y.min())
            
            print(f"\nSplit Strategy: {recommendation['strategy']}")
            print(f"Reason: {recommendation['reason']}")
            print(f"Using {recommendation['n_bins']} bins for stratification")
            
            from ai_training.split_strategy import stratified_split_regression
            X_train, X_test, y_train, y_test, split_info = stratified_split_regression(
                X, y,
                train_split=train_split,
                n_bins=recommendation['n_bins'],
                random_seed=random_seed
            )
        
        # Print split validation
        if split_info['warnings']:
            print("\n⚠️ Split Quality Warnings:")
            for warning in split_info['warnings']:
                print(f"  - {warning}")
        else:
            print("\n✅ Split is well-balanced")
        
        # Print distribution details (different for classification vs regression)
        if is_classification_mode:
            # Classification: show class distribution
            train_counts = split_info['train_distribution']['class_counts']
            test_counts = split_info['test_distribution']['class_counts']
            
            print(f"\nTrain set: {split_info['train_distribution']['count']} samples")
            for cls, count in train_counts.items():
                print(f"  Class {cls}: {count} samples")
            
            print(f"\nTest set: {split_info['test_distribution']['count']} samples")
            for cls, count in test_counts.items():
                print(f"  Class {cls}: {count} samples")
        else:
            # Regression: show statistical distribution
            print(f"\nTrain distribution: mean={split_info['train_distribution']['mean']:.3f}, "
                  f"std={split_info['train_distribution']['std']:.3f}, "
                  f"range=[{split_info['train_distribution']['min']:.3f}, {split_info['train_distribution']['max']:.3f}]")
            print(f"Test distribution:  mean={split_info['test_distribution']['mean']:.3f}, "
                  f"std={split_info['test_distribution']['std']:.3f}, "
                  f"range=[{split_info['test_distribution']['min']:.3f}, {split_info['test_distribution']['max']:.3f}]")
        
        return X_train, X_test, y_train, y_test, image_ids
    
    def prepare_augmented_training_data(
        self,
        target_feature: str,
        train_split: float = 0.8,
        random_seed: int = 42,
        augmentation_config: Optional[Dict] = None,
        output_dir: str = '/app/data/ai_training_data',
        normalizer=None
    ) -> Tuple[Dict, str]:
        """
        Prepare augmented training dataset and save to disk.
        
        Args:
            target_feature: Feature to predict
            train_split: Train/validation split ratio
            random_seed: Random seed for reproducibility
            augmentation_config: Dict with augmentation parameters:
                - rotation_range: (min, max) degrees, default: (-3, 3)
                - translation_range: (min, max) pixels, default: (-10, 10)
                - scale_range: (min, max) scale factor, default: (0.95, 1.05)
                - num_augmentations: number per image, default: 5
            output_dir: Directory to save augmented data
        
        Returns:
            (statistics_dict, output_directory_path)
        """
        from ai_training.data_augmentation import ImageAugmentor, AugmentedDatasetBuilder
        
        # Default augmentation config
        default_config = {
            'rotation_range': (-3.0, 3.0),
            'translation_range': (-10, 10),
            'scale_range': (0.95, 1.05),
            'num_augmentations': 5
        }
        
        if augmentation_config:
            default_config.update(augmentation_config)
        
        print(f"\n🔄 Preparing augmented dataset...")
        print(f"   Augmentation config: {default_config}")
        
        # Load images from database
        images = self.db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        # Filter images with target feature
        images_data = []
        
        # Check if classification
        is_classification_mode = target_feature.startswith('Custom_Class_')
        if is_classification_mode:
            num_classes_str = target_feature.replace('Custom_Class_', '')
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                
                # Check if feature exists
                has_feature = False
                if is_classification_mode:
                    if "Custom_Class" in features and num_classes_str in features.get("Custom_Class", {}):
                        has_feature = True
                else:
                    if target_feature in features:
                        has_feature = True
                
                if has_feature:
                    images_data.append({
                        'id': img.id,
                        'patient_id': img.patient_id,
                        'processed_image_data': img.processed_image_data,
                        'features_data': img.features_data
                    })
            except Exception as e:
                print(f"Warning: Error loading image {img.id}: {e}")
        
        if len(images_data) == 0:
            raise ValueError(f"No images found with feature '{target_feature}'")
        
        print(f"   Found {len(images_data)} images with feature '{target_feature}'")
        
        # Create train/val split
        y_values = []
        for img_data in images_data:
            features = json.loads(img_data['features_data'])
            
            # Extract target value
            if is_classification_mode:
                y_values.append(float(features["Custom_Class"][num_classes_str]["label"]))
            else:
                y_values.append(float(features[target_feature]))
        
        y_array = np.array(y_values)
        
        # Create stratified split indices
        indices = np.arange(len(images_data))
        
        # Choose split strategy based on task type
        if is_classification_mode:
            # For classification: stratify by class labels
            from ai_training.split_strategy import stratified_split_classification
            
            print(f"   Split strategy: stratified_classification (by class labels)")
            print(f"   Number of classes: {len(np.unique(y_array))}")
            
            _, _, _, _, split_info = stratified_split_classification(
                indices.reshape(-1, 1),
                y_array,
                train_split=train_split,
                random_seed=random_seed
            )
        else:
            # For regression: stratify by binning continuous values
            from ai_training.split_strategy import stratified_split_regression
            
            recommendation = get_split_recommendation(len(images_data), y_array.max() - y_array.min())
            
            print(f"   Split strategy: {recommendation['strategy']} with {recommendation['n_bins']} bins")
            
            _, _, _, _, split_info = stratified_split_regression(
                indices.reshape(-1, 1),
                y_array,
                train_split=train_split,
                n_bins=recommendation['n_bins'],
                random_seed=random_seed
            )
        
        # Get actual train/val indices - different logic for classification vs regression
        np.random.seed(random_seed)
        
        train_indices = []
        val_indices = []
        
        if is_classification_mode:
            # For classification: split by class labels
            unique_classes = np.unique(y_array)
            
            for cls in unique_classes:
                class_mask = y_array == cls
                class_idxs = np.where(class_mask)[0]
                np.random.shuffle(class_idxs)
                
                split_point = int(len(class_idxs) * train_split)
                
                # Ensure at least 1 sample in val if possible
                if split_point == len(class_idxs) and len(class_idxs) > 1:
                    split_point = len(class_idxs) - 1
                
                train_indices.extend(class_idxs[:split_point].tolist())
                val_indices.extend(class_idxs[split_point:].tolist())
        else:
            # For regression: split by bins
            try:
                from ai_training.split_strategy import create_bins
            except ImportError:
                from split_strategy import create_bins
            
            bin_assignments = create_bins(y_array, n_bins=recommendation['n_bins'], method='quantile')
            unique_bins = np.unique(bin_assignments)
            
            for bin_idx in unique_bins:
                bin_mask = bin_assignments == bin_idx
                bin_idxs = np.where(bin_mask)[0]
                np.random.shuffle(bin_idxs)
                
                split_point = int(len(bin_idxs) * train_split)
                train_indices.extend(bin_idxs[:split_point].tolist())
                val_indices.extend(bin_idxs[split_point:].tolist())
        
        split_indices = {
            'train': train_indices,
            'val': val_indices
        }
        
        print(f"   Train: {len(train_indices)} images, Val: {len(val_indices)} images")
        
        # Fit normalizer if provided
        if normalizer is not None:
            normalizer.fit(y_array)
            print(f"   Fitted normalizer on {len(y_array)} samples")
        
        # Create augmentor
        augmentor = ImageAugmentor(
            rotation_range=tuple(default_config['rotation_range']),
            translation_range=tuple(default_config['translation_range']),
            scale_range=tuple(default_config['scale_range']),
            num_augmentations=default_config['num_augmentations']
        )
        
        # Build augmented dataset
        builder = AugmentedDatasetBuilder(
            output_dir=output_dir,
            augmentor=augmentor,
            include_original=True,
            normalizer=normalizer
        )
        
        stats = builder.prepare_augmented_dataset(
            images_data=images_data,
            split_indices=split_indices,
            target_feature=target_feature,
            clean_existing=True
        )
        
        # Add split info to stats
        stats['split_info'] = split_info
        stats['split_strategy'] = recommendation['strategy']
        stats['n_bins'] = recommendation['n_bins']
        
        # Update metadata.json with split information
        metadata_file = Path(output_dir) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add split information
            metadata['split_strategy'] = recommendation['strategy']
            metadata['split_info'] = split_info
            metadata['n_bins'] = recommendation['n_bins']
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return stats, output_dir

