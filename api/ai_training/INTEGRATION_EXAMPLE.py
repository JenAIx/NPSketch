"""
Integration Example: How to use data augmentation in the training API

This file shows how to integrate augmented data into the existing training pipeline.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from sqlalchemy.orm import Session

from ai_training.data_loader import TrainingDataLoader
from ai_training.dataset import create_augmented_dataloaders
from ai_training.trainer import CNNTrainer
from database import get_db


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class AugmentationConfig(BaseModel):
    """Configuration for data augmentation."""
    rotation_range: tuple[float, float] = (-3.0, 3.0)
    translation_range: tuple[int, int] = (-10, 10)
    scale_range: tuple[float, float] = (0.95, 1.05)
    num_augmentations: int = 5


class TrainingConfigWithAugmentation(BaseModel):
    """Training configuration with augmentation support."""
    target_feature: str
    train_split: float = 0.8
    num_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 8
    use_augmentation: bool = True
    augmentation_config: Optional[AugmentationConfig] = None


# ============================================================
# EXAMPLE 1: Prepare Augmented Dataset Endpoint
# ============================================================

async def prepare_augmented_dataset_endpoint(
    config: TrainingConfigWithAugmentation,
    db: Session
):
    """
    Endpoint to prepare augmented dataset for training.
    
    Usage:
        POST /api/ai-training/prepare-augmented-data
        {
            "target_feature": "Total_Score",
            "train_split": 0.8,
            "use_augmentation": true,
            "augmentation_config": {
                "rotation_range": [-3, 3],
                "translation_range": [-10, 10],
                "scale_range": [0.95, 1.05],
                "num_augmentations": 5
            }
        }
    """
    try:
        # Initialize data loader
        loader = TrainingDataLoader(db)
        
        # Prepare augmentation config
        aug_config = None
        if config.use_augmentation and config.augmentation_config:
            aug_config = {
                'rotation_range': config.augmentation_config.rotation_range,
                'translation_range': config.augmentation_config.translation_range,
                'scale_range': config.augmentation_config.scale_range,
                'num_augmentations': config.augmentation_config.num_augmentations
            }
        
        # Prepare augmented dataset
        if config.use_augmentation:
            stats, output_dir = loader.prepare_augmented_training_data(
                target_feature=config.target_feature,
                train_split=config.train_split,
                augmentation_config=aug_config,
                output_dir='/app/data/ai_training_data'
            )
            
            return {
                'success': True,
                'message': 'Augmented dataset prepared successfully',
                'output_dir': output_dir,
                'statistics': stats,
                'train_samples': stats['train']['total'],
                'val_samples': stats['val']['total'],
                'augmentation_multiplier': (
                    stats['train']['total'] / stats['train']['original']
                    if stats['train']['original'] > 0 else 1
                )
            }
        else:
            return {
                'success': True,
                'message': 'Augmentation disabled, using original data',
                'use_augmentation': False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# EXAMPLE 2: Train with Augmented Data Endpoint
# ============================================================

async def train_with_augmentation_endpoint(
    config: TrainingConfigWithAugmentation,
    db: Session
):
    """
    Complete training pipeline with augmentation.
    
    Usage:
        POST /api/ai-training/train-with-augmentation
        {
            "target_feature": "Total_Score",
            "train_split": 0.8,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 8,
            "use_augmentation": true,
            "augmentation_config": {
                "rotation_range": [-3, 3],
                "translation_range": [-10, 10],
                "scale_range": [0.95, 1.05],
                "num_augmentations": 5
            }
        }
    """
    try:
        # Step 1: Prepare augmented dataset
        loader = TrainingDataLoader(db)
        
        aug_config = None
        if config.use_augmentation and config.augmentation_config:
            aug_config = {
                'rotation_range': config.augmentation_config.rotation_range,
                'translation_range': config.augmentation_config.translation_range,
                'scale_range': config.augmentation_config.scale_range,
                'num_augmentations': config.augmentation_config.num_augmentations
            }
        
        if config.use_augmentation:
            # Prepare augmented data
            stats, output_dir = loader.prepare_augmented_training_data(
                target_feature=config.target_feature,
                train_split=config.train_split,
                augmentation_config=aug_config,
                output_dir='/app/data/ai_training_data'
            )
            
            # Step 2: Create dataloaders from augmented data
            train_loader, val_loader, dataloader_stats = create_augmented_dataloaders(
                data_dir=output_dir,
                batch_size=config.batch_size,
                shuffle_train=True
            )
            
            augmentation_info = {
                'enabled': True,
                'config': aug_config,
                'train_samples': stats['train']['total'],
                'val_samples': stats['val']['total'],
                'multiplier': (
                    stats['train']['total'] / stats['train']['original']
                    if stats['train']['original'] > 0 else 1
                )
            }
        else:
            # Use original data (no augmentation)
            # This would use the existing create_dataloaders function
            # (not shown here, but available in dataset.py)
            raise HTTPException(
                status_code=400,
                detail="Non-augmented training not implemented in this example"
            )
        
        # Step 3: Initialize trainer
        trainer = CNNTrainer(
            num_outputs=1,
            learning_rate=config.learning_rate,
            device='cpu'  # or 'cuda' if GPU available
        )
        
        # Step 4: Train model
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.num_epochs
        )
        
        # Step 5: Evaluate final performance
        train_metrics = trainer.evaluate_metrics(train_loader)
        val_metrics = trainer.evaluate_metrics(val_loader)
        
        # Step 6: Save model with metadata
        model_path = trainer.save_model(
            name=f"model_{config.target_feature}",
            metadata={
                'target_feature': config.target_feature,
                'training_config': config.dict(),
                'augmentation': augmentation_info,
                'dataset': {
                    'total_samples': dataloader_stats['total_samples'],
                    'train_samples': dataloader_stats['train_samples'],
                    'val_samples': dataloader_stats['val_samples']
                },
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_architecture': 'ResNet-18'
            }
        )
        
        return {
            'success': True,
            'message': 'Training completed successfully',
            'model_path': model_path,
            'augmentation': augmentation_info,
            'training_history': training_history,
            'final_metrics': {
                'train': train_metrics,
                'validation': val_metrics
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# EXAMPLE 3: Get Augmentation Statistics
# ============================================================

async def get_augmentation_stats_endpoint():
    """
    Get statistics about prepared augmented dataset.
    
    Usage:
        GET /api/ai-training/augmentation-stats
    """
    try:
        from ai_training.data_augmentation import get_augmentation_stats
        from pathlib import Path
        
        data_dir = '/app/data/ai_training_data'
        
        if not Path(data_dir).exists():
            return {
                'success': False,
                'message': 'No augmented dataset found',
                'data_dir': data_dir
            }
        
        stats = get_augmentation_stats(data_dir)
        
        return {
            'success': True,
            'data_dir': data_dir,
            'statistics': stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# EXAMPLE 4: Integration with Existing Training Endpoint
# ============================================================

def integrate_into_existing_training_endpoint(
    existing_config: Dict,
    db: Session,
    use_augmentation: bool = True,
    augmentation_config: Optional[Dict] = None
):
    """
    Shows how to add augmentation to existing training code.
    
    This is a drop-in enhancement for the existing training endpoint
    in routers/ai_training.py
    """
    
    # Get target feature from config
    target_feature = existing_config['target_feature']
    
    if use_augmentation:
        # NEW: Prepare augmented data
        loader = TrainingDataLoader(db)
        
        stats, output_dir = loader.prepare_augmented_training_data(
            target_feature=target_feature,
            train_split=existing_config.get('train_split', 0.8),
            augmentation_config=augmentation_config or {
                'rotation_range': (-3, 3),
                'translation_range': (-10, 10),
                'scale_range': (0.95, 1.05),
                'num_augmentations': 5
            },
            output_dir='/app/data/ai_training_data'
        )
        
        # NEW: Load augmented data
        train_loader, val_loader, dataloader_stats = create_augmented_dataloaders(
            data_dir=output_dir,
            batch_size=existing_config.get('batch_size', 8)
        )
        
        # Store augmentation info for metadata
        augmentation_metadata = {
            'enabled': True,
            'config': augmentation_config,
            'stats': stats
        }
    else:
        # EXISTING: Use original data loading
        # (Your existing code here)
        pass
    
    # Rest of training code remains the same...
    # trainer = CNNTrainer(...)
    # trainer.train(train_loader, val_loader, ...)
    # etc.
    
    return {
        'augmentation_metadata': augmentation_metadata if use_augmentation else None,
        # ... rest of existing return values
    }


# ============================================================
# EXAMPLE 5: CLI Script for Testing
# ============================================================

def cli_example():
    """
    Command-line example for testing augmentation.
    
    Usage:
        python -c "from ai_training.INTEGRATION_EXAMPLE import cli_example; cli_example()"
    """
    from database import SessionLocal
    
    print("=" * 60)
    print("DATA AUGMENTATION - CLI EXAMPLE")
    print("=" * 60)
    
    # Initialize
    db = SessionLocal()
    loader = TrainingDataLoader(db)
    
    # Prepare augmented dataset
    print("\n1. Preparing augmented dataset...")
    stats, output_dir = loader.prepare_augmented_training_data(
        target_feature='Total_Score',
        train_split=0.8,
        augmentation_config={
            'rotation_range': (-3, 3),
            'translation_range': (-10, 10),
            'scale_range': (0.95, 1.05),
            'num_augmentations': 5
        },
        output_dir='/app/data/ai_training_data'
    )
    
    print(f"✅ Dataset prepared!")
    print(f"   Train: {stats['train']['total']} images")
    print(f"   Val: {stats['val']['total']} images")
    
    # Load dataloaders
    print("\n2. Creating dataloaders...")
    train_loader, val_loader, dataloader_stats = create_augmented_dataloaders(
        data_dir=output_dir,
        batch_size=8
    )
    
    print(f"✅ Dataloaders created!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Train model
    print("\n3. Training model...")
    trainer = CNNTrainer(learning_rate=0.001)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2  # Short example
    )
    
    print(f"✅ Training complete!")
    
    # Evaluate
    print("\n4. Evaluating model...")
    val_metrics = trainer.evaluate_metrics(val_loader)
    
    print(f"✅ Evaluation complete!")
    print(f"   R² Score: {val_metrics['r2_score']:.3f}")
    print(f"   RMSE: {val_metrics['rmse']:.4f}")
    print(f"   MAE: {val_metrics['mae']:.4f}")
    
    # Save model
    print("\n5. Saving model...")
    model_path = trainer.save_model(
        name="model_Total_Score",
        metadata={
            'target_feature': 'Total_Score',
            'augmentation': stats,
            'val_metrics': val_metrics
        }
    )
    
    print(f"✅ Model saved: {model_path}")
    print("\n" + "=" * 60)
    
    db.close()


if __name__ == '__main__':
    cli_example()

