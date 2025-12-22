"""
AI Training Router

Endpoints for CNN training and model management.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json
import numpy as np
import re
import sys

# Import data_loader (it will handle PyTorch imports gracefully)
sys.path.insert(0, '/app')

from ai_training.data_loader import TrainingDataLoader
from ai_training.classification_generator import generate_balanced_classes

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
        
        # 4. Pre-calculate auto-classifications (2-5 classes)
        auto_classifications = {}
        
        for num_classes in [2, 3, 4, 5]:
            try:
                result = generate_balanced_classes(scores, num_classes, method="quantile")
                auto_classifications[f"{num_classes}_classes"] = result
            except Exception as e:
                # If classification fails, skip this number
                print(f"Warning: Could not generate {num_classes} classes: {e}")
                continue
        
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
        custom_classes = config.get("custom_classes")  # Custom boundaries and names
        
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
        
        # 2. Determine class structure (custom or auto-generate)
        if method == "custom" and custom_classes:
            # Use custom classes from frontend
            class_definitions = custom_classes
            print(f"Using custom classes: {len(custom_classes)} classes")
        else:
            # Auto-generate with classification_generator
            from ai_training.classification_generator import generate_balanced_classes
            result = generate_balanced_classes(scores, num_classes, method="quantile")
            class_definitions = result["classes"]
            print(f"Auto-generated: {len(class_definitions)} classes")
        
        # Build boundaries list from class definitions
        all_boundaries = [class_definitions[0]["min"]]
        for cls in class_definitions:
            all_boundaries.append(cls["max"])
        
        # 3. Assign classes and update DB (ZUSÄTZLICH, nicht ersetzend!)
        
        updated_count = 0
        class_distribution = {i: 0 for i in range(len(class_definitions))}
        
        for img in valid_images:
            try:
                features = json.loads(img.features_data)
                score = float(features[feature_name])
                
                # Determine class based on score ranges
                class_id = None
                for cls in class_definitions:
                    if score >= cls["min"] and score <= cls["max"]:
                        class_id = cls["id"]
                        break
                
                if class_id is None:
                    # Score outside all ranges - skip
                    print(f"Warning: Score {score} not in any class range")
                    continue
                
                # Get class info
                cls_info = class_definitions[class_id]
                
                # Update features_data
                # Total_Score bleibt unverändert!
                
                # REPLACE Custom_Class completely (only one active classification!)
                num_classes_str = str(len(class_definitions))
                features["Custom_Class"] = {
                    num_classes_str: {
                        "label": int(class_id),
                        "name_custom": cls_info.get("custom_name"),
                        "name_generic": cls_info.get("generic_name") or f"Class_{class_id} [{cls_info['min']}-{cls_info['max']}]",
                        "boundaries": all_boundaries
                    }
                }
                
                img.features_data = json.dumps(features)
                class_distribution[class_id] += 1
                updated_count += 1
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error processing image {img.id}: {e}")
                continue
        
        db.commit()
        
        # Calculate class percentages
        class_info = []
        total = updated_count
        for cls_def in class_definitions:
            class_id = cls_def["id"]
            count = class_distribution.get(class_id, 0)
            percentage = (count / total * 100.0) if total > 0 else 0.0
            class_info.append({
                "class_id": class_id,
                "count": count,
                "percentage": round(percentage, 2),
                "range": f"{cls_def['min']}-{cls_def['max']}",
                "custom_name": cls_def.get("custom_name"),
                "generic_name": cls_def.get("generic_name")
            })
        
        return {
            "success": True,
            "updated_count": updated_count,
            "num_classes": len(class_definitions),
            "actual_num_classes": len(class_definitions),
            "method": method,
            "boundaries": all_boundaries,
            "class_distribution": class_distribution,
            "class_info": class_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating classes: {str(e)}")


@router.get("/custom-class-distribution/{feature_name}")
async def get_custom_class_distribution(
    feature_name: str,
    db: Session = Depends(get_db)
):
    """
    Get distribution for an existing Custom_Class feature.
    
    Args:
        feature_name: e.g., "Custom_Class_5"
    
    Returns:
        Class distribution with counts and percentages
    """
    try:
        # Extract num_classes from feature name
        if not feature_name.startswith("Custom_Class_"):
            raise HTTPException(status_code=400, detail="Invalid feature name")
        
        num_classes_str = feature_name.replace("Custom_Class_", "")
        
        # Get all images with this classification
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        class_counts = {}
        class_info = {}
        total_samples = 0
        
        for img in images:
            try:
                features = json.loads(img.features_data)
                
                # Check if Custom_Class exists
                if "Custom_Class" not in features:
                    continue
                
                # Check if this num_classes exists
                if num_classes_str not in features["Custom_Class"]:
                    continue
                
                class_data = features["Custom_Class"][num_classes_str]
                class_id = class_data["label"]
                
                # Count this class
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                    class_info[class_id] = {
                        "name_custom": class_data.get("name_custom"),
                        "name_generic": class_data.get("name_generic"),
                        "boundaries": class_data.get("boundaries")
                    }
                
                class_counts[class_id] += 1
                total_samples += 1
                
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        if total_samples == 0:
            raise HTTPException(status_code=404, detail="No samples found with this classification")
        
        # Build class list with percentages
        classes = []
        boundaries = class_info[0]["boundaries"] if 0 in class_info else []
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_samples) * 100.0
            info = class_info[class_id]
            
            # Extract range from boundaries or generic name
            if boundaries and len(boundaries) > class_id + 1:
                # Calculate non-overlapping range for this class
                range_min = boundaries[class_id]
                range_max = boundaries[class_id + 1] - 1 if class_id < len(boundaries) - 2 else boundaries[class_id + 1]
                
                range_str = f"[{range_min}, {range_max}]" if range_min != range_max else f"= {range_min}"
            else:
                # Parse from generic name as fallback
                match = re.search(r'\[([^\]]+)\]', info["name_generic"] or "")
                range_str = match.group(0) if match else f"Class {class_id}"
            
            classes.append({
                "id": class_id,
                "name_custom": info["name_custom"],
                "name_generic": info["name_generic"],
                "range": range_str,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return {
            "feature_name": feature_name,
            "num_classes": num_classes_str,
            "total_samples": total_samples,
            "classes": classes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading custom class distribution: {str(e)}")


@router.post("/recalculate-classes")
async def recalculate_class_counts(
    config: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Recalculate class counts based on custom boundaries.
    
    Request Body:
    {
        "feature_name": "Total_Score",
        "num_classes": 5,
        "boundaries": [[2, 42], [43, 50], [51, 58], [59, 59], [60, 60]]
    }
    """
    try:
        feature_name = config.get("feature_name")
        num_classes = config.get("num_classes")
        boundaries = config.get("boundaries")  # List of [min, max] pairs
        
        if not feature_name or not boundaries:
            raise HTTPException(status_code=400, detail="feature_name and boundaries required")
        
        # Get all scores
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        scores = []
        for img in images:
            try:
                features = json.loads(img.features_data)
                if feature_name in features:
                    scores.append(float(features[feature_name]))
            except:
                continue
        
        if len(scores) == 0:
            raise HTTPException(status_code=404, detail="No scores found")
        
        scores = np.array(scores)
        total = len(scores)
        
        # Calculate counts for each class
        result_classes = []
        for class_id, (class_min, class_max) in enumerate(boundaries):
            # Count scores in this range
            mask = (scores >= class_min) & (scores <= class_max)
            count = int(np.sum(mask))
            percentage = (count / total) * 100.0
            
            result_classes.append({
                "id": class_id,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return {
            "success": True,
            "classes": result_classes,
            "total": total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recalculating: {str(e)}")


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
