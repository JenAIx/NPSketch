"""
AI Training Router

Endpoints for CNN training and model management.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json

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
            'images_data': images_data
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
        
        from ai_training.dataset import create_dataloaders
        from ai_training.trainer import CNNTrainer
        
        # Create dataloaders
        train_loader, val_loader, stats = create_dataloaders(
            config['images_data'],
            config['target_feature'],
            train_split=config['train_split'],
            batch_size=config['batch_size']
        )
        
        training_state['progress']['message'] = f"Loaded {stats['total_samples']} samples"
        
        # Create trainer
        trainer = CNNTrainer(
            num_outputs=1,
            learning_rate=config['learning_rate']
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
        
        # Prepare metadata
        metadata = {
            'target_feature': config['target_feature'],
            'training_config': {
                'train_split': config['train_split'],
                'num_epochs': config['num_epochs'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size']
            },
            'dataset': {
                'total_samples': stats['total_samples'],
                'train_samples': stats['train_samples'],
                'val_samples': stats['val_samples'],
                'train_batches': stats['train_batches'],
                'val_batches': stats['val_batches']
            },
            'model_architecture': 'ResNet-18',
            'input_size': '568x274x1',
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_history': trainer.history,
            'image_ids': [img['id'] for img in config['images_data'] if 'id' in img],
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
    """Delete a saved model."""
    import os
    from pathlib import Path
    
    model_path = Path("/app/data/models") / model_filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        os.unlink(model_path)
        return {"success": True, "message": f"Model '{model_filename}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

