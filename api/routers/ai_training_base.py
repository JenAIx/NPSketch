"""
AI Training Base Router

Core endpoints for training management and dataset info.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, TrainingDataImage
from pydantic import ValidationError
import json
import sys
from datetime import datetime

sys.path.insert(0, '/app')
from ai_training.data_loader import TrainingDataLoader
from config.models import TrainingConfig, TrainingStartResponse, TrainingStatus
from utils.logger import get_logger, setup_from_config
from config import get_config

# Setup logging
config = get_config()
setup_from_config(config)  # Pass full config, not just logging section
logger = get_logger(__name__)

router = APIRouter(prefix="/api/ai-training", tags=["ai_training_base"])

# Global training state
training_state = {
    'status': 'idle',  # idle, training, completed, error, cancelled
    'progress': {},
    'error': None,
    'cancelled': False  # Flag to request cancellation
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


@router.post("/start-training", response_model=TrainingStartResponse)
async def start_training(config_dict: dict = Body(...), db: Session = Depends(get_db)):
    """Start CNN training (async, returns immediately) with type-safe configuration."""
    import threading
    
    try:
        # Validate configuration with Pydantic
        try:
            training_config = TrainingConfig(**config_dict)
            logger.info(f"Training request validated: target={training_config.target_feature}, "
                       f"epochs={training_config.num_epochs}, augmentation={training_config.use_augmentation}")
        except ValidationError as e:
            logger.error(f"Invalid training configuration: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        target_feature = training_config.target_feature
        train_split = training_config.train_split
        num_epochs = training_config.num_epochs
        learning_rate = training_config.learning_rate
        batch_size = training_config.batch_size
        use_augmentation = training_config.use_augmentation
        use_normalization = training_config.use_normalization
        add_synthetic_bad_images = training_config.add_synthetic_bad_images
        synthetic_n_samples = training_config.synthetic_n_samples
        
        # Validation is now handled by Pydantic, but keep for backwards compatibility
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
            'add_synthetic_bad_images': add_synthetic_bad_images,
            'synthetic_n_samples': synthetic_n_samples,
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
                'add_synthetic_bad_images': add_synthetic_bad_images,
                'synthetic_n_samples': synthetic_n_samples,
                'total_samples': len(images_data)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_training_job(config):
    """Run training in background thread with structured logging."""
    global training_state
    
    logger.info("="*60)
    logger.info(f"TRAINING JOB STARTED")
    logger.info(f"Target Feature: {config['target_feature']}")
    logger.info(f"Epochs: {config['num_epochs']}, Batch Size: {config['batch_size']}")
    logger.info(f"Augmentation: {config.get('use_augmentation', True)}")
    logger.info(f"Synthetic Images: {config.get('add_synthetic_bad_images', False)} "
               f"(n={config.get('synthetic_n_samples', 0)})")
    logger.info("="*60)
    
    try:
        import time
        start_time = time.time()
        
        # Reset cancellation flag when starting new training
        training_state['cancelled'] = False
        training_state['status'] = 'training'
        training_state['progress'] = {
            'epoch': 0,
            'total_epochs': config['num_epochs'],
            'train_loss': 0,
            'val_loss': 0,
            'message': 'Initializing...',
            'start_time': start_time,
            'training_config': {
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'train_split': config['train_split'],
                'use_augmentation': config.get('use_augmentation', True),
                'use_normalization': config.get('use_normalization', True)
            }
        }
        
        logger.debug(f"Training state initialized: {training_state}")
        
        from ai_training.trainer import CNNTrainer
        from ai_training.normalization import get_normalizer_for_feature
        
        use_normalization = config.get('use_normalization', True)
        use_augmentation = config.get('use_augmentation', True)
        target_feature = config['target_feature']
        
        # Detect if this is classification or regression
        is_classification = target_feature.startswith('Custom_Class_')
        num_classes = None  # Initialize for regression case
        
        if is_classification:
            # Extract num_classes from feature name (e.g., "Custom_Class_5" -> 5)
            num_classes = int(target_feature.replace('Custom_Class_', ''))
            num_outputs = num_classes
            training_mode = "classification"
            normalizer = None  # NO normalization for classification!
            
            # Update training_config to reflect actual state (normalization is always disabled for classification)
            training_state['progress']['training_config']['use_normalization'] = False
            
            logger.info(f"Training mode: CLASSIFICATION ({num_classes} classes)")
            logger.info(f"Output neurons: {num_outputs}")
            logger.info("Target normalization: Disabled (classification uses raw class indices)")
        else:
            # Regression mode
            num_outputs = 1
            training_mode = "regression"
            
            normalizer = None
            actual_normalization_enabled = False
            if use_normalization:
                normalizer = get_normalizer_for_feature(target_feature)
                actual_normalization_enabled = (normalizer is not None)
            
            # Update training_config to reflect actual state
            training_state['progress']['training_config']['use_normalization'] = actual_normalization_enabled
            
            logger.info("Training mode: REGRESSION")
            logger.info(f"Output neurons: {num_outputs}")
            if normalizer:
                logger.info("Target normalization: Enabled")
        
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
                    normalizer=normalizer,
                    add_synthetic_bad_images=config.get('add_synthetic_bad_images', False),
                    synthetic_n_samples=config.get('synthetic_n_samples', 50)
                )
                
                train_loader, val_loader, stats = create_augmented_dataloaders(
                    data_dir=output_dir,
                    batch_size=config['batch_size'],
                    shuffle_train=True,
                    is_classification=is_classification
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
                
                # Transfer synthetic_bad_images info from aug_stats to stats
                stats['synthetic_bad_images'] = aug_stats.get('synthetic_bad_images', {'enabled': False})
                
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
                normalizer=normalizer,
                is_classification=is_classification,
                num_classes=num_classes if is_classification else None
            )
            
            stats['augmentation'] = {'enabled': False}
        
        # Remove image IDs from stats for progress display (not needed for status info)
        # Image IDs are still saved in model metadata for later use
        stats_for_progress = stats.copy()
        stats_for_progress.pop('train_image_ids', None)
        stats_for_progress.pop('val_image_ids', None)
        
        training_state['progress']['dataset_stats'] = stats_for_progress
        training_state['progress']['training_mode'] = training_mode
        training_state['progress']['num_classes'] = num_classes if is_classification else None
        
        trainer = CNNTrainer(
            num_outputs=num_outputs,
            learning_rate=config['learning_rate'],
            normalizer=normalizer,
            training_mode=training_mode
        )
        
        epoch_times = []  # Track time per epoch for estimation
        
        for epoch in range(config['num_epochs']):
            # Check for cancellation request
            if training_state.get('cancelled', False):
                logger.info("Training cancellation requested by user")
                training_state['status'] = 'cancelled'
                training_state['progress']['message'] = 'Training cancelled by user'
                training_state['error'] = 'Training was cancelled by user'
                return
            
            epoch_start_time = time.time()
            
            training_state['progress']['epoch'] = epoch + 1
            training_state['progress']['message'] = f"Training epoch {epoch+1}/{config['num_epochs']}..."
            
            metrics = trainer.train_epoch(train_loader, val_loader)
            
            # Update training history
            trainer.history['epoch'].append(epoch)
            trainer.history['train_loss'].append(metrics['train_loss'])
            if metrics['val_loss'] is not None:
                trainer.history['val_loss'].append(metrics['val_loss'])
            
            # Calculate duration and estimated remaining time
            current_time = time.time()
            elapsed_time = current_time - start_time
            epoch_duration = current_time - epoch_start_time
            epoch_times.append(epoch_duration)
            
            # Calculate estimated remaining time based on average epoch time
            estimated_remaining = 0
            if len(epoch_times) > 0:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = config['num_epochs'] - (epoch + 1)
                estimated_remaining = avg_epoch_time * remaining_epochs
            
            training_state['progress']['train_loss'] = metrics['train_loss']
            training_state['progress']['val_loss'] = metrics['val_loss']
            training_state['progress']['duration_seconds'] = int(elapsed_time)
            training_state['progress']['estimated_remaining_seconds'] = int(estimated_remaining)
        
        train_metrics = trainer.evaluate_metrics(train_loader)
        val_metrics = trainer.evaluate_metrics(val_loader)
        
        from ai_training.model import get_model_summary
        model_info = get_model_summary(trainer.model)
        
        metadata = {
            'target_feature': config['target_feature'],
            'training_mode': training_mode,
            'num_classes': num_classes,
            'use_sigmoid': trainer.use_sigmoid,  # Store sigmoid usage for model loading
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
            'synthetic_bad_images': stats.get('synthetic_bad_images', {'enabled': False}),
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


@router.post("/stop-training")
async def stop_training():
    """Stop/cancel the current training job."""
    global training_state
    
    if training_state['status'] != 'training':
        raise HTTPException(
            status_code=400, 
            detail=f"No training job is currently running. Current status: {training_state['status']}"
        )
    
    # Set cancellation flag
    training_state['cancelled'] = True
    logger.info("Training cancellation requested via API")
    
    return {
        "success": True,
        "message": "Training cancellation requested. The job will stop after the current epoch completes."
    }


@router.get("/training-status")
async def get_training_status():
    """Get current training status and progress."""
    import time
    global training_state
    
    # Calculate current duration if training is in progress
    if training_state['status'] == 'training' and 'progress' in training_state:
        progress = training_state['progress']
        if 'start_time' in progress:
            current_time = time.time()
            elapsed_time = current_time - progress['start_time']
            progress['duration_seconds'] = int(elapsed_time)
            
            # Update estimated remaining if we have epoch info
            if 'epoch' in progress and 'total_epochs' in progress:
                epoch = progress.get('epoch', 0)
                total_epochs = progress.get('total_epochs', 0)
                
                if epoch > 0 and total_epochs > 0:
                    # Estimate based on current progress
                    avg_time_per_epoch = elapsed_time / epoch if epoch > 0 else 0
                    remaining_epochs = total_epochs - epoch
                    estimated_remaining = avg_time_per_epoch * remaining_epochs
                    progress['estimated_remaining_seconds'] = int(estimated_remaining)
    
    return training_state

