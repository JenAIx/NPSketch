#!/usr/bin/env python3
"""
Analyze the latest Custom_Class_4 training results.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, '/app')

def find_latest_model():
    """Find the latest Custom_Class_4 model."""
    model_dir = Path('/app/data/models')
    pattern = 'model_Custom_Class_4_*.json'
    models = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0] if models else None

def compare_models(old_path, new_path):
    """Compare two model metadata files."""
    with open(old_path, 'r') as f:
        old = json.load(f)
    with open(new_path, 'r') as f:
        new = json.load(f)
    
    print("="*70)
    print("MODEL COMPARISON: Old vs New")
    print("="*70)
    
    # Config comparison
    print("\n📋 Configuration Comparison:")
    old_tc = old.get('training_config', {})
    new_tc = new.get('training_config', {})
    
    config_items = [
        ('learning_rate', 'Learning Rate'),
        ('dropout', 'Dropout'),
        ('weight_decay', 'Weight Decay'),
        ('label_smoothing', 'Label Smoothing'),
        ('warmup_enabled', 'Warmup Enabled'),
        ('warmup_epochs', 'Warmup Epochs'),
        ('lr_scheduling_patience', 'LR Scheduling Patience'),
        ('lr_scheduling_threshold', 'LR Scheduling Threshold'),
        ('early_stopping_min_epochs', 'Early Stopping Min Epochs'),
    ]
    
    for key, label in config_items:
        old_val = old_tc.get(key, 'N/A')
        new_val = new_tc.get(key, 'N/A')
        status = "✅" if old_val != new_val or key in ['warmup_enabled', 'label_smoothing'] else "  "
        print(f"  {label:30} {old_val:15} → {new_val:15} {status}")
    
    # Performance comparison
    print("\n🎯 Performance Comparison:")
    
    old_train = old.get('train_metrics', {})
    old_val = old.get('val_metrics', {})
    new_train = new.get('train_metrics', {})
    new_val = new.get('val_metrics', {})
    
    print(f"\n  Validation Accuracy:")
    old_acc = old_val.get('accuracy', 0) * 100
    new_acc = new_val.get('accuracy', 0) * 100
    diff = new_acc - old_acc
    print(f"    Old: {old_acc:.2f}%")
    print(f"    New: {new_acc:.2f}%")
    print(f"    Change: {diff:+.2f}% {'✅' if diff > 0 else '❌'}")
    
    print(f"\n  Validation Macro F1:")
    old_f1 = old_val.get('macro_f1', 0)
    new_f1 = new_val.get('macro_f1', 0)
    diff = new_f1 - old_f1
    print(f"    Old: {old_f1:.4f}")
    print(f"    New: {new_f1:.4f}")
    print(f"    Change: {diff:+.4f} {'✅' if diff > 0 else '❌'}")
    
    print(f"\n  Overfitting (Train-Val Gap):")
    old_gap = (old_train.get('accuracy', 0) - old_val.get('accuracy', 0)) * 100
    new_gap = (new_train.get('accuracy', 0) - new_val.get('accuracy', 0)) * 100
    print(f"    Old: {old_gap:.2f}%")
    print(f"    New: {new_gap:.2f}%")
    print(f"    Change: {new_gap - old_gap:+.2f}% {'✅' if new_gap < old_gap else '❌'}")
    
    # Early stopping
    print(f"\n⏹️  Early Stopping:")
    old_es = old.get('early_stopping', {})
    new_es = new.get('early_stopping', {})
    print(f"  Best Epoch:")
    print(f"    Old: {old_es.get('best_epoch', 'N/A')}")
    print(f"    New: {new_es.get('best_epoch', 'N/A')} {'✅' if new_es.get('best_epoch', 1) > 1 else '⚠️'}")
    print(f"  Total Epochs:")
    print(f"    Old: {old_es.get('total_epochs_trained', 'N/A')}")
    print(f"    New: {new_es.get('total_epochs_trained', 'N/A')}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    # Find latest model
    latest = find_latest_model()
    if not latest:
        print("❌ No Custom_Class_4 models found")
        sys.exit(1)
    
    print(f"📊 Analyzing latest model: {latest.name}\n")
    
    # Load and display
    with open(latest, 'r') as f:
        data = json.load(f)
    
    print("="*70)
    print("LATEST TRAINING RESULTS")
    print("="*70)
    print(f"\nModel: {latest.name}")
    print(f"Trained at: {data.get('trained_at', 'N/A')}")
    
    # Config
    tc = data.get('training_config', {})
    print(f"\n📋 Configuration:")
    print(f"  Learning rate: {tc.get('learning_rate', 'N/A')}")
    print(f"  Dropout: {tc.get('dropout', 'N/A')}")
    print(f"  Weight decay: {tc.get('weight_decay', 'N/A')}")
    print(f"  Label smoothing: {tc.get('label_smoothing', 'N/A')}")
    print(f"  Warmup: {tc.get('warmup_enabled', False)} ({tc.get('warmup_epochs', 0)} epochs)")
    print(f"  LR scheduling: patience={tc.get('lr_scheduling_patience', 'N/A')}, threshold={tc.get('lr_scheduling_threshold', 'N/A')}")
    
    # Early stopping
    es = data.get('early_stopping', {})
    print(f"\n⏹️  Early Stopping:")
    print(f"  Best epoch: {es.get('best_epoch', 'N/A')} {'⚠️ Epoch 1!' if es.get('best_epoch') == 1 else '✅'}")
    print(f"  Best val_loss: {es.get('best_val_loss', 0):.6f}")
    print(f"  Total epochs: {es.get('total_epochs_trained', 'N/A')}")
    print(f"  Min epochs: {es.get('min_epochs', 'N/A')}")
    
    # Performance
    train_metrics = data.get('train_metrics', {})
    val_metrics = data.get('val_metrics', {})
    
    print(f"\n🎯 Performance:")
    print(f"  Train Accuracy: {train_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Train F1: {train_metrics.get('macro_f1', 0):.4f}")
    print(f"  Val Accuracy: {val_metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Val F1: {val_metrics.get('macro_f1', 0):.4f}")
    
    gap = (train_metrics.get('accuracy', 0) - val_metrics.get('accuracy', 0)) * 100
    print(f"  Overfitting Gap: {gap:.2f}%")
    
    # Compare with old model if available
    old_model = Path('/app/data/models/model_Custom_Class_4_20260114_121708_metadata.json')
    if old_model.exists():
        print("\n")
        compare_models(old_model, latest)
    else:
        print("\n(No old model found for comparison)")
