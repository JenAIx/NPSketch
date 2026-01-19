#!/usr/bin/env python3
"""
Test script to validate task-specific config loading for classification.
"""
import sys
sys.path.insert(0, '/app')

from routers.ai_training_base import get_task_specific_config
from config import get_config

def test_config_loading():
    """Test that classification config is loaded correctly."""
    print("="*60)
    print("TEST: Task-Specific Config Loading")
    print("="*60)
    
    # Simulate user config (minimal, should use defaults from YAML)
    user_config = {
        'target_feature': 'Custom_Class_4',
        'num_epochs': 10,  # Test with 10 epochs
        'batch_size': 8
    }
    
    # Test classification config
    print("\n1. Testing CLASSIFICATION config loading...")
    classification_config = get_task_specific_config('classification', user_config)
    
    print("\nLoaded Classification Config:")
    print(f"  Learning rate: {classification_config['learning_rate']} (expected: 0.0005)")
    print(f"  Dropout: {classification_config['dropout']} (expected: 0.6)")
    print(f"  Weight decay: {classification_config['weight_decay']} (expected: 0.0005)")
    print(f"  Label smoothing: {classification_config['label_smoothing']} (expected: 0.1)")
    print(f"  Warmup enabled: {classification_config['warmup_enabled']} (expected: True)")
    print(f"  Warmup epochs: {classification_config['warmup_epochs']} (expected: 3)")
    print(f"  LR scheduling patience: {classification_config['lr_scheduling_patience']} (expected: 3)")
    print(f"  LR scheduling threshold: {classification_config['lr_scheduling_threshold']} (expected: 0.01)")
    print(f"  Early stopping min_epochs: {classification_config['early_stopping_min_epochs']} (expected: 3)")
    
    # Validate
    errors = []
    if classification_config['learning_rate'] != 0.0005:
        errors.append(f"Learning rate mismatch: {classification_config['learning_rate']} != 0.0005")
    if classification_config['dropout'] != 0.6:
        errors.append(f"Dropout mismatch: {classification_config['dropout']} != 0.6")
    if classification_config['weight_decay'] != 0.0005:
        errors.append(f"Weight decay mismatch: {classification_config['weight_decay']} != 0.0005")
    if classification_config['label_smoothing'] != 0.1:
        errors.append(f"Label smoothing mismatch: {classification_config['label_smoothing']} != 0.1")
    if not classification_config['warmup_enabled']:
        errors.append(f"Warmup not enabled: {classification_config['warmup_enabled']} != True")
    if classification_config['warmup_epochs'] != 3:
        errors.append(f"Warmup epochs mismatch: {classification_config['warmup_epochs']} != 3")
    if classification_config['lr_scheduling_patience'] != 3:
        errors.append(f"LR scheduling patience mismatch: {classification_config['lr_scheduling_patience']} != 3")
    if classification_config['lr_scheduling_threshold'] != 0.01:
        errors.append(f"LR scheduling threshold mismatch: {classification_config['lr_scheduling_threshold']} != 0.01")
    if classification_config['early_stopping_min_epochs'] != 3:
        errors.append(f"Early stopping min_epochs mismatch: {classification_config['early_stopping_min_epochs']} != 3")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All classification config values are correct!")
    
    # Test regression config for comparison
    print("\n2. Testing REGRESSION config loading...")
    regression_config = get_task_specific_config('regression', user_config)
    
    print("\nLoaded Regression Config:")
    print(f"  Learning rate: {regression_config['learning_rate']} (expected: 0.001)")
    print(f"  Dropout: {regression_config['dropout']} (expected: 0.5)")
    print(f"  Weight decay: {regression_config['weight_decay']} (expected: 0.0001)")
    print(f"  LR scheduling patience: {regression_config['lr_scheduling_patience']} (expected: 5)")
    print(f"  Early stopping min_epochs: {regression_config['early_stopping_min_epochs']} (expected: 2)")
    
    print("\n✅ Config loading test completed!")
    return True

if __name__ == '__main__':
    success = test_config_loading()
    sys.exit(0 if success else 1)
