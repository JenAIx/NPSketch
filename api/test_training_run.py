#!/usr/bin/env python3
"""
Test script to run a short training job with Custom_Class_4 to validate config loading.
"""
import sys
sys.path.insert(0, '/app')

from database import SessionLocal, TrainingDataImage
from routers.ai_training_base import run_training_job
import json

def test_training():
    """Run a short test training."""
    print("="*60)
    print("TEST: Starting Classification Training (5 epochs)")
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
            print("❌ Not enough images for training (need at least 10)")
            return False
        
        # Prepare images data
        images_data = []
        for img in valid_images[:100]:  # Limit to 100 for quick test
            images_data.append({
                'id': img.id,
                'processed_image_data': img.processed_image_data,
                'features_data': img.features_data
            })
        
        # Training config - minimal, should use classification defaults
        config = {
            'target_feature': 'Custom_Class_4',
            'train_split': 0.8,
            'num_epochs': 5,  # Short test
            'learning_rate': None,  # Should use default 0.0005
            'batch_size': 8,
            'use_augmentation': True,
            'use_normalization': False,  # Classification doesn't use normalization
            'add_synthetic_bad_images': False,  # Skip for quick test
            'synthetic_n_samples': 0,
            'use_lr_scheduling': None,  # Should use default True
            'use_differential_lr': None,  # Should use default True
            'backbone_lr_multiplier': None,  # Should use default 0.1
            'dropout': None,  # Should use default 0.6
            'weight_decay': None,  # Should use default 0.0005
            'early_stopping_patience': None,  # Should use default 10
            'early_stopping_min_delta': None,  # Should use default 0.01
            'images_data': images_data,
            'db_session': db
        }
        
        print("\nStarting training...")
        print("Expected config values:")
        print("  Learning rate: 0.0005")
        print("  Dropout: 0.6")
        print("  Weight decay: 0.0005")
        print("  Label smoothing: 0.1")
        print("  Warmup: enabled=True, epochs=3")
        print("  LR scheduling: patience=3, threshold=0.01")
        print("  Early stopping: min_epochs=3")
        print("\n" + "="*60)
        
        # Run training
        run_training_job(config)
        
        print("\n" + "="*60)
        print("✅ Training test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == '__main__':
    success = test_training()
    sys.exit(0 if success else 1)
