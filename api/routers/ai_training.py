"""
AI Training Router

Endpoints for CNN training and model management.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json
import numpy as np

# Import data_loader (it will handle PyTorch imports gracefully)
# We need to add ai_training to path first
import sys
sys.path.insert(0, '/app')

from ai_training.data_loader import TrainingDataLoader

router = APIRouter(prefix="/api/ai-training", tags=["ai_training"])


@router.get("/dataset-info")
async def get_dataset_info(db: Session = Depends(get_db)):
    """
    Get information about available training data.
    
    Returns:
        Dataset statistics
    """
    try:
        loader = TrainingDataLoader(db)
        info = loader.get_dataset_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-features")
async def get_available_features(db: Session = Depends(get_db)):
    """
    Get all available feature labels with statistics.
    
    Returns:
        List of features with count, mean, std, min, max
    """
    try:
        loader = TrainingDataLoader(db)
        features_info = loader.get_available_features()
        return features_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """
    Get detailed information about the CNN model architecture.
    
    Returns:
        Model architecture details, parameters, layers
    """
    from ai_training.model import DrawingClassifier, get_model_summary
    
    try:
        # Create a dummy model to get architecture info
        model = DrawingClassifier(num_outputs=1, pretrained=False)
        summary = get_model_summary(model)
        
        # Add detailed layer information
        detailed_info = {
            **summary,
            "layers": {
                "input_layer": {
                    "type": "Conv2d (modified for grayscale)",
                    "channels": "1 → 64",
                    "kernel": "7×7",
                    "stride": 2,
                    "padding": 3
                },
                "backbone": {
                    "architecture": "ResNet-18",
                    "blocks": "4 residual blocks",
                    "layers": [
                        "Layer 1: 64 channels",
                        "Layer 2: 128 channels",
                        "Layer 3: 256 channels",
                        "Layer 4: 512 channels"
                    ],
                    "pretrained": "ImageNet (adapted from RGB to grayscale)"
                },
                "output_head": {
                    "layer_1": "Linear(512 → 256)",
                    "activation_1": "ReLU",
                    "dropout": "Dropout(p=0.5)",
                    "layer_2": "Linear(256 → num_outputs)",
                    "output": "Single value (regression)"
                }
            },
            "training_details": {
                "optimizer": "Adam",
                "loss_function": "MSE (Mean Squared Error)",
                "input_preprocessing": [
                    "Grayscale conversion",
                    "Normalization to [0, 1]",
                    "Size: 568×274 pixels"
                ],
                "data_augmentation": "None (currently)",
                "regularization": [
                    "Dropout (0.5)",
                    "Batch Normalization (in ResNet blocks)"
                ]
            }
        }
        
        return detailed_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-readiness")
async def check_training_readiness(db: Session = Depends(get_db)):
    """
    Check if system is ready for training.
    
    Returns:
        Readiness status with recommendations
    """
    try:
        loader = TrainingDataLoader(db)
        info = loader.get_dataset_info()
        features_info = loader.get_available_features()
        
        # Check requirements
        ready = True
        issues = []
        recommendations = []
        
        # Minimum data requirements (relaxed for testing)
        if info['total_images'] < 2:
            ready = False
            issues.append(f"Only {info['total_images']} images (minimum 2 required for train/test split)")
            recommendations.append("Upload more training data")
        
        if info['labeled_images'] < 2:
            ready = False
            issues.append(f"Only {info['labeled_images']} labeled images (minimum 2 required)")
            recommendations.append("Add features to more images")
        
        if not features_info['features']:
            ready = False
            issues.append("No features defined")
            recommendations.append("Add feature labels to training data")
        
        # Warnings (not blocking)
        if info['total_images'] < 10:
            issues.append(f"Only {info['total_images']} images (10+ recommended for better results)")
        
        # Check feature distribution
        for feature in features_info['features']:
            stats = features_info['stats'][feature]
            if stats['count'] < 2:
                ready = False
                issues.append(f"Feature '{feature}' only has {stats['count']} samples (minimum 2 required)")
                recommendations.append(f"Add more samples with '{feature}' label")
        
        return {
            "ready": ready,
            "status": "ready" if ready else "not_ready",
            "issues": issues,
            "recommendations": recommendations,
            "dataset_info": info,
            "features": features_info['features']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-training")
async def start_training(
    config: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Start CNN training (async, returns immediately).
    
    Training runs in background, progress available via /training-status
    """
    from fastapi import BackgroundTasks
    import threading
    
    try:
        # Extract parameters from config
        target_feature = config.get('target_feature')
        train_split = config.get('train_split', 0.8)
        num_epochs = config.get('num_epochs', 10)
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 8)
        use_augmentation = config.get('use_augmentation', True)  # Enabled by default
        use_normalization = config.get('use_normalization', True)  # Enabled by default
        
        # Validate parameters
        if not target_feature:
            raise HTTPException(status_code=400, detail="target_feature is required")
        
        if train_split <= 0 or train_split >= 1:
            raise HTTPException(status_code=400, detail="train_split must be between 0 and 1")
        
        if num_epochs < 1 or num_epochs > 1000:
            raise HTTPException(status_code=400, detail="num_epochs must be between 1 and 1000")
        
        # Load images
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        images_data = []
        for img in images:
            images_data.append({
                'id': img.id,
                'processed_image_data': img.processed_image_data,
                'features_data': img.features_data
            })
        
        # Start training in thread
        training_config = {
            'target_feature': target_feature,
            'train_split': train_split,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'use_augmentation': use_augmentation,
            'use_normalization': use_normalization,
            'images_data': images_data,
            'db_session': db  # Pass db session for augmentation
        }
        
        # Store in global state for progress tracking
        import threading
        training_thread = threading.Thread(
            target=run_training_job,
            args=(training_config,),
            daemon=True
        )
        training_thread.start()
        
        return {
            "success": True,
            "message": "Training started",
            "config": {
                'target_feature': target_feature,
                'train_split': train_split,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'use_augmentation': use_augmentation,
                'use_normalization': use_normalization,
                'total_samples': len(images_data)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Global training state
training_state = {
    'status': 'idle',  # idle, training, completed, error
    'progress': {},
    'error': None
}


def run_training_job(config):
    """Run training in background thread."""
    global training_state
    
    from datetime import datetime
    
    try:
        training_state['status'] = 'training'
        training_state['progress'] = {
            'epoch': 0,
            'total_epochs': config['num_epochs'],
            'train_loss': 0,
            'val_loss': 0,
            'message': 'Initializing...'
        }
        
        from ai_training.trainer import CNNTrainer
        from ai_training.normalization import get_normalizer_for_feature
        
        # Check if normalization and augmentation are enabled
        use_normalization = config.get('use_normalization', True)
        use_augmentation = config.get('use_augmentation', True)
        
        # Create normalizer if enabled
        normalizer = None
        if use_normalization:
            normalizer = get_normalizer_for_feature(config['target_feature'])
            print(f"   Target normalization: Enabled for {config['target_feature']}")
        else:
            print(f"   Target normalization: Disabled (using raw values)")
        
        if use_augmentation:
            # Use data augmentation
            training_state['progress']['message'] = 'Preparing augmented dataset...'
            
            from ai_training.data_loader import TrainingDataLoader
            from ai_training.dataset import create_augmented_dataloaders
            
            # Get a new database session (thread-safe)
            from database import SessionLocal
            db = SessionLocal()
            
            try:
                # Prepare augmented dataset (with normalizer if enabled)
                loader = TrainingDataLoader(db)
                aug_stats, output_dir = loader.prepare_augmented_training_data(
                    target_feature=config['target_feature'],
                    train_split=config['train_split'],
                    augmentation_config={
                        'rotation_range': (-3, 3),
                        'translation_range': (-10, 10),
                        'scale_range': (0.95, 1.05),
                        'num_augmentations': 5
                    },
                    output_dir='/app/data/ai_training_data',
                    normalizer=normalizer
                )
                
                training_state['progress']['message'] = f"Augmented dataset prepared: {aug_stats['train']['total']} train, {aug_stats['val']['total']} val"
                
                # Create dataloaders from augmented data
                train_loader, val_loader, stats = create_augmented_dataloaders(
                    data_dir=output_dir,
                    batch_size=config['batch_size'],
                    shuffle_train=True
                )
                
                # Add augmentation info to stats
                stats['augmentation'] = {
                    'enabled': True,
                    'original_samples': aug_stats['train']['original'] + aug_stats['val']['original'],
                    'augmented_samples': aug_stats['train']['augmented'] + aug_stats['val']['augmented'],
                    'total_samples': aug_stats['train']['total'] + aug_stats['val']['total'],
                    'multiplier': round((aug_stats['train']['total'] + aug_stats['val']['total']) / 
                                      (aug_stats['train']['original'] + aug_stats['val']['original']), 1) 
                                   if (aug_stats['train']['original'] + aug_stats['val']['original']) > 0 else 1,
                    'config': aug_stats.get('augmentation_config', {})
                }
                
            finally:
                db.close()
        else:
            # Use regular dataset (no augmentation)
            training_state['progress']['message'] = 'Loading dataset...'
            
            from ai_training.dataset import create_dataloaders
            
            train_loader, val_loader, stats = create_dataloaders(
                config['images_data'],
                config['target_feature'],
                train_split=config['train_split'],
                batch_size=config['batch_size'],
                random_seed=42,
                normalizer=normalizer
            )
            
            stats['augmentation'] = {
                'enabled': False
            }
        
        training_state['progress']['message'] = f"Loaded {stats['total_samples']} samples (augmentation: {'enabled' if use_augmentation else 'disabled'})"
        training_state['progress']['split_info'] = stats.get('split_info', {})
        training_state['progress']['split_strategy'] = stats.get('split_strategy', 'unknown')
        training_state['progress']['dataset_stats'] = stats  # Store complete stats for later
        
        # Create trainer with normalizer
        trainer = CNNTrainer(
            num_outputs=1,
            learning_rate=config['learning_rate'],
            normalizer=normalizer
        )
        
        # Training loop
        for epoch in range(config['num_epochs']):
            training_state['progress']['epoch'] = epoch + 1
            training_state['progress']['message'] = f"Training epoch {epoch+1}/{config['num_epochs']}..."
            
            # Train epoch
            metrics = trainer.train_epoch(train_loader, val_loader)
            
            training_state['progress']['train_loss'] = metrics['train_loss']
            training_state['progress']['val_loss'] = metrics['val_loss']
        
        # Final evaluation with comprehensive metrics
        training_state['progress']['message'] = 'Evaluating model performance...'
        
        train_metrics = trainer.evaluate_metrics(train_loader)
        val_metrics = trainer.evaluate_metrics(val_loader)
        
        # Get model information
        from ai_training.model import get_model_summary
        model_info = get_model_summary(trainer.model)
        
        # Prepare metadata
        metadata = {
            'target_feature': config['target_feature'],
            'training_config': {
                'train_split': config['train_split'],
                'num_epochs': config['num_epochs'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'use_augmentation': config.get('use_augmentation', True),
                'use_normalization': config.get('use_normalization', True)
            },
            'normalization': normalizer.get_config() if normalizer is not None else {'enabled': False},
            'augmentation': stats.get('augmentation', {'enabled': False}),
            'dataset': {
                'total_samples': stats['total_samples'],
                'train_samples': stats['train_samples'],
                'val_samples': stats['val_samples'],
                'train_batches': stats['train_batches'],
                'val_batches': stats['val_batches'],
                'split_strategy': stats.get('split_strategy', 'unknown'),
                'n_bins': stats.get('n_bins', 0),
                'train_target_range': stats.get('train_target_range', []),
                'val_target_range': stats.get('val_target_range', [])
            },
            'split_quality': stats.get('split_info', {}),
            'model': model_info,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': trainer.history,
            'train_image_ids': stats.get('train_image_ids', []),
            'val_image_ids': stats.get('val_image_ids', []),
            'all_image_ids': [img['id'] for img in config['images_data'] if 'id' in img],
            'trained_at': datetime.now().isoformat()
        }
        
        # Save model with metadata
        model_path = trainer.save_model(f"model_{config['target_feature']}", metadata=metadata)
        
        training_state['status'] = 'completed'
        training_state['progress']['message'] = 'Training completed!'
        training_state['progress']['model_path'] = model_path
        training_state['progress']['train_metrics'] = train_metrics
        training_state['progress']['val_metrics'] = val_metrics
        
    except Exception as e:
        training_state['status'] = 'error'
        training_state['error'] = str(e)
        import traceback
        training_state['progress']['message'] = f'Error: {str(e)}'
        traceback.print_exc()


@router.get("/training-status")
async def get_training_status():
    """Get current training status and progress."""
    global training_state
    return training_state


@router.get("/models")
async def list_models():
    """List all saved models with metadata."""
    import os
    from pathlib import Path
    
    models_dir = Path("/app/data/models")
    models_dir.mkdir(exist_ok=True)
    
    models = []
    for model_file in models_dir.glob("*.pth"):
        stat = model_file.stat()
        
        # Extract info from filename: model_{feature}_{timestamp}.pth
        parts = model_file.stem.split('_')
        if len(parts) >= 3:
            feature = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]
            timestamp = '_'.join(parts[-2:])
        else:
            feature = "unknown"
            timestamp = "unknown"
        
        # Check if metadata file exists
        metadata_file = models_dir / f"{model_file.stem}_metadata.json"
        has_metadata = metadata_file.exists()
        
        model_info = {
            'filename': model_file.name,
            'path': str(model_file),
            'feature': feature,
            'timestamp': timestamp,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_at': stat.st_mtime,
            'has_metadata': has_metadata
        }
        
        models.append(model_info)
    
    # Sort by creation time (newest first)
    models.sort(key=lambda x: x['created_at'], reverse=True)
    
    return {
        'models': models,
        'total': len(models)
    }


@router.get("/models/{model_filename}/metadata")
async def get_model_metadata(model_filename: str):
    """Get metadata for a specific model."""
    from pathlib import Path
    import json
    
    # Remove .pth extension if present
    model_stem = model_filename.replace('.pth', '')
    metadata_file = Path("/app/data/models") / f"{model_stem}_metadata.json"
    
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/test")
async def test_model(
    request: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Test a saved model on current test data.
    
    Returns comprehensive evaluation metrics.
    """
    import os
    from pathlib import Path
    import torch
    from ai_training.trainer import CNNTrainer
    from ai_training.dataset import create_dataloaders
    
    try:
        model_filename = request.get('model_filename')
        if not model_filename:
            raise HTTPException(status_code=400, detail="model_filename is required")
        
        model_path = Path("/app/data/models") / model_filename
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Load model checkpoint to get feature info
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract feature from filename
        parts = model_filename.split('_')
        feature = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]
        
        # Load training data
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        images_data = []
        for img in images:
            images_data.append({
                'id': img.id,
                'processed_image_data': img.processed_image_data,
                'features_data': img.features_data
            })
        
        # Create dataloaders (same split as training for consistency)
        train_loader, val_loader, stats = create_dataloaders(
            images_data,
            feature,
            train_split=0.8,
            batch_size=8
        )
        
        # Create trainer and load model
        trainer = CNNTrainer(num_outputs=1)
        trainer.load_model(str(model_path))
        
        # Evaluate on both sets
        train_metrics = trainer.evaluate_metrics(train_loader)
        val_metrics = trainer.evaluate_metrics(val_loader)
        
        return {
            'success': True,
            'model': model_filename,
            'target_feature': feature,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'dataset_stats': stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_filename}")
async def delete_model(model_filename: str):
    """Delete a saved model and its metadata."""
    import os
    from pathlib import Path
    
    model_path = Path("/app/data/models") / model_filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Delete model file
        os.unlink(model_path)
        
        # Delete metadata file if exists
        model_stem = model_filename.replace('.pth', '')
        metadata_path = Path("/app/data/models") / f"{model_stem}_metadata.json"
        
        if metadata_path.exists():
            os.unlink(metadata_path)
            return {"success": True, "message": f"Model and metadata deleted: {model_filename}"}
        
        return {"success": True, "message": f"Model deleted: {model_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-distribution/{feature_name}")
async def get_feature_distribution(
    feature_name: str,
    db: Session = Depends(get_db)
):
    """
    Get distribution data for a feature (histogram, stats, auto-classifications).
    Returns only aggregated data, not individual values.
    
    Args:
        feature_name: Name of the feature (e.g., 'Total_Score')
    
    Returns:
        Distribution data with histogram bins, statistics, and pre-calculated class splits
    """
    try:
        # 1. Fetch all scores from DB
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        scores = []
        for img in images:
            try:
                features = json.loads(img.features_data)
                if feature_name in features:
                    score = float(features[feature_name])
                    scores.append(score)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        if len(scores) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No samples found with feature '{feature_name}'"
            )
        
        scores = np.array(scores)
        
        # 2. Calculate statistics
        stats = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "range": float(np.max(scores) - np.min(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75))
        }
        
        # 3. Generate histogram (25 bins)
        hist_counts, hist_edges = np.histogram(scores, bins=25)
        histogram_data = []
        total_samples = len(scores)
        
        for i in range(len(hist_counts)):
            bin_min = float(hist_edges[i])
            bin_max = float(hist_edges[i + 1])
            count = int(hist_counts[i])
            percentage = (count / total_samples) * 100.0
            
            histogram_data.append({
                "min": bin_min,
                "max": bin_max,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        histogram = {
            "bins": 25,
            "data": histogram_data
        }
        
        # 4. Pre-calculate auto-classifications (2-5 classes, quantile-based)
        auto_classifications = {}
        
        for num_classes in [2, 3, 4, 5]:
            # Calculate quantile-based boundaries
            percentiles = np.linspace(0, 100, num_classes + 1)
            boundaries = np.percentile(scores, percentiles)
            
            # Ensure boundaries are unique and sorted
            boundaries = np.unique(boundaries)
            if len(boundaries) < num_classes + 1:
                # Fallback to equal-width if quantiles produce duplicates
                boundaries = np.linspace(np.min(scores), np.max(scores), num_classes + 1)
            
            # Assign classes and count distribution
            class_data = []
            
            for class_id in range(num_classes):
                class_min = float(boundaries[class_id])
                class_max = float(boundaries[class_id + 1])
                
                # Count samples in this class
                if class_id == num_classes - 1:
                    # Last class: include upper boundary
                    mask = (scores >= class_min) & (scores <= class_max)
                else:
                    mask = (scores >= class_min) & (scores < class_max)
                
                count = int(np.sum(mask))
                percentage = (count / total_samples) * 100.0
                
                class_data.append({
                    "id": class_id,
                    "min": class_min,
                    "max": class_max,
                    "count": count,
                    "percentage": round(percentage, 2),
                    "label": f"Class_{class_id}"
                })
            
            auto_classifications[f"{num_classes}_classes"] = {
                "method": "quantile",
                "boundaries": [float(b) for b in boundaries],
                "classes": class_data
            }
        
        return {
            "feature_name": feature_name,
            "total_samples": total_samples,
            "statistics": stats,
            "histogram": histogram,
            "auto_classifications": auto_classifications
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating distribution: {str(e)}")


@router.post("/generate-classes")
async def generate_and_save_classes(
    config: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Generate class boundaries and save class labels to database.
    
    Request Body:
    {
        "feature_name": "Total_Score",
        "num_classes": 4,
        "method": "quantile"  # or "equal-width"
    }
    
    Returns:
        Success status and class distribution
    """
    try:
        feature_name = config.get("feature_name")
        num_classes = config.get("num_classes")
        method = config.get("method", "quantile")
        
        if not feature_name:
            raise HTTPException(status_code=400, detail="feature_name is required")
        
        if not num_classes or num_classes < 2 or num_classes > 10:
            raise HTTPException(
                status_code=400,
                detail="num_classes must be between 2 and 10"
            )
        
        # 1. Get all scores
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        scores = []
        valid_images = []
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                if feature_name in features:
                    score = float(features[feature_name])
                    scores.append(score)
                    valid_images.append(img)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        if len(scores) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No samples found with feature '{feature_name}'"
            )
        
        scores = np.array(scores)
        
        # 2. Calculate class boundaries
        if method == "quantile":
            percentiles = np.linspace(0, 100, num_classes + 1)
            boundaries = np.percentile(scores, percentiles)
            # Ensure boundaries are unique
            boundaries = np.unique(boundaries)
            if len(boundaries) < num_classes + 1:
                # Fallback to equal-width if quantiles produce duplicates
                boundaries = np.linspace(np.min(scores), np.max(scores), num_classes + 1)
        else:  # equal-width
            boundaries = np.linspace(np.min(scores), np.max(scores), num_classes + 1)
        
        boundaries = np.array(boundaries)
        
        # 3. Assign classes and update DB (ZUSÄTZLICH, nicht ersetzend!)
        updated_count = 0
        class_distribution = {i: 0 for i in range(num_classes)}
        
        for img in valid_images:
            try:
                features = json.loads(img.features_data)
                score = float(features[feature_name])
                
                # Determine class (digitize returns bin index, 0-based)
                class_id = np.digitize(score, boundaries) - 1
                # Clamp to valid range
                class_id = max(0, min(int(class_id), num_classes - 1))
                
                # Update features_data - NUR HINZUFÜGEN!
                # Total_Score bleibt unverändert!
                features[f"class_label_{num_classes}_classes"] = int(class_id)
                
                # Create class name with range
                class_min = float(boundaries[class_id])
                class_max = float(boundaries[class_id + 1])
                features[f"class_name_{num_classes}_classes"] = f"Class_{class_id} ({class_min:.1f}-{class_max:.1f})"
                
                # Optional: Store boundaries for reference
                features[f"classification_boundaries_{num_classes}_classes"] = [float(b) for b in boundaries]
                
                img.features_data = json.dumps(features)
                class_distribution[class_id] += 1
                updated_count += 1
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                continue
        
        db.commit()
        
        # Calculate class percentages
        class_info = []
        total = updated_count
        for class_id in range(num_classes):
            count = class_distribution[class_id]
            percentage = (count / total * 100.0) if total > 0 else 0.0
            class_info.append({
                "class_id": class_id,
                "count": count,
                "percentage": round(percentage, 2),
                "range": f"{float(boundaries[class_id]):.1f}-{float(boundaries[class_id + 1]):.1f}"
            })
        
        return {
            "success": True,
            "updated_count": updated_count,
            "num_classes": num_classes,
            "method": method,
            "boundaries": [float(b) for b in boundaries],
            "class_distribution": class_distribution,
            "class_info": class_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating classes: {str(e)}")


@router.post("/cleanup-orphaned-metadata")
async def cleanup_orphaned_metadata():
    """Delete orphaned metadata files (metadata without corresponding .pth file)."""
    from pathlib import Path
    import os
    
    models_dir = Path("/app/data/models")
    cleaned = 0
    
    try:
        # Find all metadata files
        for metadata_file in models_dir.glob("*_metadata.json"):
            # Check if corresponding .pth file exists
            model_stem = metadata_file.stem.replace('_metadata', '')
            model_file = models_dir / f"{model_stem}.pth"
            
            if not model_file.exists():
                # Orphaned metadata - delete it
                os.unlink(metadata_file)
                cleaned += 1
        
        return {
            "success": True,
            "cleaned": cleaned,
            "message": f"Deleted {cleaned} orphaned metadata file(s)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
