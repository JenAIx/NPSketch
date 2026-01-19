#!/usr/bin/env python3
"""
Start a full training run with Custom_Class_4 to validate all fixes.
"""
import sys
sys.path.insert(0, '/app')

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job
import json
import time

def start_training():
    """Start a full training run."""
    print("="*60)
    print("Starting FULL Classification Training")
    print("="*60)
    
    db = SessionLocal()
    try:
        # Load images with Custom_Class_4
        images = db.query(TrainingDataImage).filter(
            TrainingDataImage.features_data.isnot(None)
        ).all()
        
        # Filter for images with Custom_Class_4
        valid_images = []
        for img in images:
            try:
                features = json.loads(img.features_data)
                if 'Custom_Class' in features:
                    custom_class = features['Custom_Class']
                    if '4' in custom_class:
                        valid_images.append(img)
            except (json.JSONDecodeError, KeyError):
                continue
        
        print(f"\nFound {len(valid_images)} images with Custom_Class_4")
        
        if len(valid_images) < 10:
            print("❌ Not enough images for training")
            return False
        
        # Prepare images data (use all valid images)
        images_data = []
        for img in valid_images:
            images_data.append({
                'id': img.id,
                'processed_image_data': img.processed_image_data,
                'features_data': img.features_data
            })
        
        # Training config - use defaults from YAML (None values will use task-specific defaults)
        config = {
            'target_feature': 'Custom_Class_4',
            'train_split': 0.8,
            'num_epochs': 30,  # Full training
            'learning_rate': None,  # Will use 0.0005 from classification config
            'batch_size': 8,
            'use_augmentation': True,
            'use_normalization': False,  # Classification doesn't use normalization
            'add_synthetic_bad_images': False,  # Skip for now
            'synthetic_n_samples': 0,
            'use_lr_scheduling': None,  # Will use default True
            'use_differential_lr': None,  # Will use default True
            'backbone_lr_multiplier': None,  # Will use default 0.1
            'dropout': None,  # Will use 0.6 from classification config
            'weight_decay': None,  # Will use 0.0005 from classification config
            'early_stopping_patience': None,  # Will use 10 from classification config
            'early_stopping_min_delta': None,  # Will use 0.01 from classification config
            'images_data': images_data,
            'db_session': db
        }
        
        print("\nTraining Configuration:")
        print("  Target: Custom_Class_4")
        print("  Epochs: 30")
        print("  Using task-specific defaults from training_config.yaml")
        print("\nExpected Classification Config:")
        print("  Learning rate: 0.0005")
        print("  Dropout: 0.6")
        print("  Weight decay: 0.0005")
        print("  Label smoothing: 0.1")
        print("  Warmup: enabled=True, epochs=3")
        print("  LR scheduling: patience=3, threshold=0.01")
        print("  Early stopping: patience=10, min_delta=0.01, min_epochs=3")
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")
        
        # Run training (this will take a while)
        start_time = time.time()
        run_training_job(config)
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"✅ Training completed in {elapsed/60:.1f} minutes")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == '__main__':
    success = start_training()
    sys.exit(0 if success else 1)
