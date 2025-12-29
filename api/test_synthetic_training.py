"""
Test Script: Train with Synthetic Bad Images

Tests both Classification and Regression modes with synthetic bad images enabled.
"""

import sys
import json
from database import SessionLocal
from ai_training.data_loader import TrainingDataLoader
from ai_training.normalization import get_normalizer_for_feature
from ai_training.dataset import create_augmented_dataloaders
from ai_training.trainer import CNNTrainer


def test_classification():
    """Test Classification mode with synthetic bad images."""
    print('=' * 70)
    print('TEST 1: CLASSIFICATION (Custom_Class_4) mit Synthetic Bad Images')
    print('=' * 70)
    
    db = SessionLocal()
    
    try:
        loader = TrainingDataLoader(db)
        
        # Check if Custom_Class_4 exists
        available = loader.get_available_features()
        if 'Custom_Class_4' not in available['features']:
            print('❌ Custom_Class_4 not found in database')
            print(f'Available features: {available["features"]}')
            return False
        
        print('✅ Custom_Class_4 found in database')
        
        # Prepare augmented data WITH synthetic bad images
        print('\n📊 Preparing augmented dataset with synthetic bad images...')
        
        aug_stats, output_dir = loader.prepare_augmented_training_data(
            target_feature='Custom_Class_4',
            train_split=0.8,
            random_seed=42,
            augmentation_config={'num_augmentations': 5},
            output_dir='/app/data/ai_training_data_test_class',
            normalizer=None,  # No normalizer for classification
            add_synthetic_bad_images=True,
            synthetic_n_samples=25
        )
        
        print(f'\n✅ Dataset prepared:')
        print(f'   Train: {aug_stats["train"]["total"]} images')
        print(f'   Val: {aug_stats["val"]["total"]} images')
        
        # Create dataloaders
        print('\n📊 Creating dataloaders...')
        train_loader, val_loader, dl_stats = create_augmented_dataloaders(
            data_dir=output_dir,
            batch_size=8,
            shuffle_train=True,
            is_classification=True
        )
        
        print(f'   Train batches: {len(train_loader)}')
        print(f'   Val batches: {len(val_loader)}')
        
        # Initialize trainer
        print('\n🚀 Starting training (5 epochs)...')
        
        num_classes = 4
        trainer = CNNTrainer(
            num_outputs=num_classes,
            learning_rate=0.001,
            training_mode='classification'
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5
        )
        
        print('\n✅ Training completed!')
        print(f'   Final train loss: {history["train_loss"][-1]:.4f}')
        print(f'   Final val loss: {history["val_loss"][-1]:.4f}')
        
        # Evaluate
        print('\n📊 Evaluating model...')
        metrics = trainer.evaluate_metrics(val_loader)
        print(f'   Accuracy: {metrics["accuracy"]:.3f}')
        print(f'   Macro F1: {metrics["macro_f1"]:.3f}')
        
        # Save model
        trainer.save_model(
            name='test_class_with_synthetic',
            metadata={
                'target_feature': 'Custom_Class_4',
                'training_mode': 'classification',
                'num_classes': num_classes,
                'synthetic_bad_images': True,
                'synthetic_n_samples': 25,
                'test_run': True
            }
        )
        
        print('\n✅ Test 1 (Classification) completed successfully!')
        return True
        
    except Exception as e:
        import traceback
        print(f'\n❌ Error: {e}')
        print(traceback.format_exc())
        return False
    
    finally:
        db.close()


def test_regression():
    """Test Regression mode with synthetic bad images."""
    print('\n\n')
    print('=' * 70)
    print('TEST 2: REGRESSION (Total_Score) mit Synthetic Bad Images')
    print('=' * 70)
    
    db = SessionLocal()
    
    try:
        loader = TrainingDataLoader(db)
        
        # Prepare augmented data WITH synthetic bad images
        print('\n📊 Preparing augmented dataset with synthetic bad images...')
        
        normalizer = get_normalizer_for_feature('Total_Score')
        
        aug_stats, output_dir = loader.prepare_augmented_training_data(
            target_feature='Total_Score',
            train_split=0.8,
            random_seed=42,
            augmentation_config={'num_augmentations': 5},
            output_dir='/app/data/ai_training_data_test_regr',
            normalizer=normalizer,
            add_synthetic_bad_images=True,
            synthetic_n_samples=25
        )
        
        print(f'\n✅ Dataset prepared:')
        print(f'   Train: {aug_stats["train"]["total"]} images')
        print(f'   Val: {aug_stats["val"]["total"]} images')
        
        # Create dataloaders
        print('\n📊 Creating dataloaders...')
        train_loader, val_loader, dl_stats = create_augmented_dataloaders(
            data_dir=output_dir,
            batch_size=8,
            shuffle_train=True,
            is_classification=False
        )
        
        print(f'   Train batches: {len(train_loader)}')
        print(f'   Val batches: {len(val_loader)}')
        
        # Initialize trainer
        print('\n🚀 Starting training (5 epochs)...')
        
        trainer = CNNTrainer(
            num_outputs=1,
            learning_rate=0.001,
            normalizer=normalizer,
            training_mode='regression'
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5
        )
        
        print('\n✅ Training completed!')
        print(f'   Final train loss: {history["train_loss"][-1]:.4f}')
        print(f'   Final val loss: {history["val_loss"][-1]:.4f}')
        
        # Evaluate
        print('\n📊 Evaluating model...')
        metrics = trainer.evaluate_metrics(val_loader)
        print(f'   R² Score: {metrics["r2_score"]:.3f}')
        print(f'   RMSE: {metrics["rmse"]:.2f}')
        print(f'   MAE: {metrics["mae"]:.2f}')
        
        # Save model
        trainer.save_model(
            name='test_regr_with_synthetic',
            metadata={
                'target_feature': 'Total_Score',
                'training_mode': 'regression',
                'synthetic_bad_images': True,
                'synthetic_n_samples': 25,
                'test_run': True,
                'normalization': normalizer.get_config()
            }
        )
        
        print('\n✅ Test 2 (Regression) completed successfully!')
        return True
        
    except Exception as e:
        import traceback
        print(f'\n❌ Error: {e}')
        print(traceback.format_exc())
        return False
    
    finally:
        db.close()


if __name__ == '__main__':
    # Run both tests
    test1_success = test_classification()
    test2_success = test_regression()
    
    print('\n\n')
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Test 1 (Classification): {"✅ PASSED" if test1_success else "❌ FAILED"}')
    print(f'Test 2 (Regression): {"✅ PASSED" if test2_success else "❌ FAILED"}')
    
    if test1_success and test2_success:
        print('\n🎉 All tests completed successfully!')
    else:
        print('\n⚠️ Some tests failed. Check logs above.')

