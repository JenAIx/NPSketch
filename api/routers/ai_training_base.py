"""
AI Training Base Router

Core endpoints for training management and dataset info.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
import json
import sys
from datetime import datetime

sys.path.insert(0, '/app')
from ai_training.data_loader import TrainingDataLoader

router = APIRouter(prefix="/api/ai-training", tags=["ai_training_base"])

# Global training state
training_state = {
    'status': 'idle',  # idle, training, completed, error
    'progress': {},
    'error': None
}


@router.get("/dataset-info")
async def get_dataset_info(db: Session = Depends(get_db)):
    """Get information about available training data."""
    try:
        loader = TrainingDataLoader(db)
        info = loader.get_dataset_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-features")
async def get_available_features(db: Session = Depends(get_db)):
    """Get all available feature labels with statistics."""
    try:
        loader = TrainingDataLoader(db)
        features_info = loader.get_available_features()
        return features_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get detailed information about the CNN model architecture."""
    from ai_training.model import DrawingClassifier, get_model_summary
    
    try:
        model = DrawingClassifier(num_outputs=1, pretrained=False)
        summary = get_model_summary(model)
        
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
    """Check if system is ready for training."""
    try:
        loader = TrainingDataLoader(db)
        info = loader.get_dataset_info()
        features_info = loader.get_available_features()
        
        ready = True
        issues = []
        recommendations = []
        
        if info['total_images'] < 2:
            ready = False
            issues.append(f"Only {info['total_images']} images (minimum 2 required)")
            recommendations.append("Upload more training data")
        
        if info['labeled_images'] < 2:
            ready = False
            issues.append(f"Only {info['labeled_images']} labeled images")
            recommendations.append("Add features to more images")
        
        if not features_info['features']:
            ready = False
            issues.append("No features defined")
            recommendations.append("Add feature labels to training data")
        
        if info['total_images'] < 10:
            issues.append(f"Only {info['total_images']} images (10+ recommended)")
        
        for feature in features_info['features']:
            stats = features_info['stats'][feature]
            if stats['count'] < 2:
                ready = False
                issues.append(f"Feature '{feature}' only has {stats['count']} samples")
        
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
async def start_training(config: dict = Body(...), db: Session = Depends(get_db)):
    """Start CNN training (async, returns immediately)."""
    import threading
    
    try:
        target_feature = config.get('target_feature')
        train_split = config.get('train_split', 0.8)
        num_epochs = config.get('num_epochs', 10)
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 8)
        use_augmentation = config.get('use_augmentation', True)
        use_normalization = config.get('use_normalization', True)
        
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
        
        training_config = {
            'target_feature': target_feature,
            'train_split': train_split,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'use_augmentation': use_augmentation,
            'use_normalization': use_normalization,
            'images_data': images_data,
            'db_session': db
        }
        
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


def run_training_job(config):
    """Run training in background thread."""
    global training_state
    
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
        
        use_normalization = config.get('use_normalization', True)
        use_augmentation = config.get('use_augmentation', True)
        
        normalizer = None
        if use_normalization:
            normalizer = get_normalizer_for_feature(config['target_feature'])
        
        if use_augmentation:
            training_state['progress']['message'] = 'Preparing augmented dataset...'
            
            from ai_training.data_loader import TrainingDataLoader
            from ai_training.dataset import create_augmented_dataloaders
            from database import SessionLocal
            
            db = SessionLocal()
            
            try:
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
                
                train_loader, val_loader, stats = create_augmented_dataloaders(
                    data_dir=output_dir,
                    batch_size=config['batch_size'],
                    shuffle_train=True
                )
                
                stats['augmentation'] = {
                    'enabled': True,
                    'original_samples': aug_stats['train']['original'] + aug_stats['val']['original'],
                    'augmented_samples': aug_stats['train']['augmented'] + aug_stats['val']['augmented'],
                    'total_samples': aug_stats['train']['total'] + aug_stats['val']['total'],
                    'multiplier': round((aug_stats['train']['total'] + aug_stats['val']['total']) / 
                                      (aug_stats['train']['original'] + aug_stats['val']['original']), 1),
                    'config': aug_stats.get('augmentation_config', {})
                }
                
            finally:
                db.close()
        else:
            from ai_training.dataset import create_dataloaders
            
            train_loader, val_loader, stats = create_dataloaders(
                config['images_data'],
                config['target_feature'],
                train_split=config['train_split'],
                batch_size=config['batch_size'],
                random_seed=42,
                normalizer=normalizer
            )
            
            stats['augmentation'] = {'enabled': False}
        
        training_state['progress']['dataset_stats'] = stats
        
        trainer = CNNTrainer(
            num_outputs=1,
            learning_rate=config['learning_rate'],
            normalizer=normalizer
        )
        
        for epoch in range(config['num_epochs']):
            training_state['progress']['epoch'] = epoch + 1
            training_state['progress']['message'] = f"Training epoch {epoch+1}/{config['num_epochs']}..."
            
            metrics = trainer.train_epoch(train_loader, val_loader)
            
            training_state['progress']['train_loss'] = metrics['train_loss']
            training_state['progress']['val_loss'] = metrics['val_loss']
        
        train_metrics = trainer.evaluate_metrics(train_loader)
        val_metrics = trainer.evaluate_metrics(val_loader)
        
        from ai_training.model import get_model_summary
        model_info = get_model_summary(trainer.model)
        
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
            'normalization': normalizer.get_config() if normalizer else {'enabled': False},
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

