#!/usr/bin/env python3
"""
Simple test - Only test data loading (no PyTorch required).
"""

import sys
sys.path.insert(0, '/app')

from database import SessionLocal

# Direct import to avoid PyTorch dependency in __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("data_loader", "/app/ai_training/data_loader.py")
data_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader_module)
TrainingDataLoader = data_loader_module.TrainingDataLoader

import json


def main():
    print("="*80)
    print("SIMPLE TEST - Data Loading Only")
    print("="*80)
    
    db = SessionLocal()
    loader = TrainingDataLoader(db)
    
    # Test 1: Get dataset info
    print("\nTest 1: Dataset Info")
    print("-"*80)
    info = loader.get_dataset_info()
    
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    if info['total_images'] == 0:
        print("\n⚠️ No training data in database")
        print("   Upload some data via http://localhost/ai_training_data_upload.html")
        db.close()
        return False
    
    # Test 2: Get available features
    print("\nTest 2: Available Features")
    print("-"*80)
    features_info = loader.get_available_features()
    
    if not features_info['features']:
        print("⚠️ No features found")
        print("   Add features via:")
        print("   1. Manual: Click entries in ai_training_data_view.html")
        print("   2. Bulk: Upload CSV with features")
        db.close()
        return False
    
    print(f"Available features: {features_info['features']}")
    print("\nFeature Statistics:")
    
    for feature in features_info['features']:
        stats = features_info['stats'][feature]
        print(f"  {feature}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean:  {stats['mean']:.3f}")
        print(f"    Std:   {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Test 3: Load numpy data
    if features_info['features']:
        print("\nTest 3: Load Training Data (NumPy)")
        print("-"*80)
        
        target_feature = features_info['features'][0]
        print(f"Target feature: {target_feature}")
        
        try:
            X_train, X_test, y_train, y_test, image_ids = loader.load_training_data(
                target_feature,
                train_split=0.8,
                random_seed=42
            )
            
            print(f"\nTrain set:")
            print(f"  Images (X_train): {X_train.shape}")
            print(f"  Labels (y_train): {y_train.shape}")
            print(f"  Value range: [{y_train.min():.3f}, {y_train.max():.3f}]")
            
            print(f"\nTest set:")
            print(f"  Images (X_test): {X_test.shape}")
            print(f"  Labels (y_test): {y_test.shape}")
            print(f"  Value range: [{y_test.min():.3f}, {y_test.max():.3f}]")
            
            print(f"\nImage IDs: {image_ids}")
            
            print("\n✅ All tests PASSED!")
            db.close()
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            db.close()
            return False
    
    db.close()
    return True


if __name__ == '__main__':
    success = main()
    print("\n" + "="*80)
    if success:
        print("SUCCESS: Data loading works correctly")
    else:
        print("FAILED: Check errors above")
    print("="*80)
    sys.exit(0 if success else 1)

