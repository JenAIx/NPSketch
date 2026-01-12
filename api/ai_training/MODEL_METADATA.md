# Model Metadata Structure

This document describes the complete structure of the model metadata JSON that is saved with each trained model.

## Complete Metadata Structure

```json
{
  "target_feature": "Custom_Class_Total_Score",
  "training_mode": "classification | regression",
  "num_classes": 4,
  "use_sigmoid": false,
  
  "training_config": {
    "train_split": 0.8,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 8,
    "use_augmentation": true,
    "use_normalization": true,
    "use_lr_scheduling": true,
    "use_differential_lr": true,
    "backbone_lr_multiplier": 0.1,
    "dropout": 0.5,
    "weight_decay": 0.0001,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001
  },
  
  "regularization": {
    "dropout": 0.5,
    "weight_decay": 0.0001
  },
  
  "early_stopping": {
    "enabled": true,
    "patience": 15,
    "min_delta": 0.001,
    "triggered": false,
    "stopped_epoch": null,
    "best_epoch": 35,
    "best_val_loss": 0.1234,
    "total_epochs_trained": 50
  },
  
  "lr_scheduling": {
    "enabled": true,
    "strategy": "ReduceLROnPlateau",
    "factor": 0.5,
    "patience": 5,
    "min_lr": 0.000001,
    "final_lr": 0.000125
  },
  
  "differential_lr": {
    "enabled": true,
    "backbone_multiplier": 0.1,
    "backbone_lr": 0.0001,
    "head_lr": 0.001
  },
  
  "class_weights": {
    "enabled": true,
    "weights": [10.4459, 9.5826, 4.7520, 3.9040]
  },
  
  "normalization": {
    "enabled": true,
    "method": "min_max",
    "train_min": 0.0,
    "train_max": 60.0,
    "feature_name": "Total_Score"
  },
  
  "augmentation": {
    "enabled": true,
    "num_augmentations": 5,
    "rotation_range": [-3, 3],
    "translation_range": [-10, 10],
    "scale_range": [0.95, 1.05],
    "use_warping": true,
    "warping_displacement": 15,
    "warping_ratio": 0.4
  },
  
  "synthetic_bad_images": {
    "enabled": false,
    "n_samples": 0,
    "n_generated": 0
  },
  
  "model": {
    "name": "DrawingClassifier",
    "architecture": "ResNet-18",
    "backbone": "ResNet-18 (ImageNet pre-trained)",
    "input_size": "568×274×1",
    "input_channels": 1,
    "output_neurons": 4,
    "dropout": 0.5,
    "use_sigmoid": false,
    "total_parameters": 11689988,
    "trainable_parameters": 11689988,
    "frozen_parameters": 0,
    "head_layers": [
      {"type": "Linear", "in_features": 512, "out_features": 256, "parameters": 131328},
      {"type": "ReLU"},
      {"type": "Dropout", "p": 0.5},
      {"type": "Linear", "in_features": 256, "out_features": 4, "parameters": 1028}
    ],
    "pretrained_weights": "ImageNet1K_V1",
    "framework": "PyTorch"
  },
  
  "dataset": {
    "total_samples": 200,
    "train_samples": 160,
    "val_samples": 40,
    "train_batches": 20,
    "val_batches": 5,
    "split_strategy": "stratified_classification",
    "n_bins": 0,
    "train_target_range": [],
    "val_target_range": []
  },
  
  "split_quality": {
    "method": "stratified_classification",
    "train_distribution": {
      "count": 160,
      "class_counts": {
        "0": 23,
        "1": 24,
        "2": 49,
        "3": 64
      }
    },
    "test_distribution": {
      "count": 40,
      "class_counts": {
        "0": 5,
        "1": 6,
        "2": 12,
        "3": 17
      }
    },
    "warnings": []
  },
  
  "train_metrics": {
    "loss": 0.1234,
    "accuracy": 0.95,
    "f1_score_macro": 0.94,
    "per_class": {
      "class_0": {"precision": 0.92, "recall": 0.91, "f1": 0.915, "support": 23},
      "class_1": {"precision": 0.93, "recall": 0.94, "f1": 0.935, "support": 24},
      "class_2": {"precision": 0.95, "recall": 0.96, "f1": 0.955, "support": 49},
      "class_3": {"precision": 0.97, "recall": 0.98, "f1": 0.975, "support": 64}
    },
    "confusion_matrix": [[21, 2, 0, 0], [1, 22, 1, 0], [0, 1, 47, 1], [0, 0, 1, 63]]
  },
  
  "val_metrics": {
    "loss": 0.2345,
    "accuracy": 0.90,
    "f1_score_macro": 0.89,
    "per_class": {
      "class_0": {"precision": 0.85, "recall": 0.83, "f1": 0.84, "support": 5},
      "class_1": {"precision": 0.88, "recall": 0.90, "f1": 0.89, "support": 6},
      "class_2": {"precision": 0.91, "recall": 0.92, "f1": 0.915, "support": 12},
      "class_3": {"precision": 0.94, "recall": 0.95, "f1": 0.945, "support": 17}
    },
    "confusion_matrix": [[4, 1, 0, 0], [0, 5, 1, 0], [0, 1, 11, 0], [0, 0, 1, 16]]
  },
  
  "training_history": {
    "epoch": [0, 1, 2, 3, ..., 49],
    "train_loss": [1.2, 0.8, 0.5, 0.3, ..., 0.12],
    "val_loss": [1.5, 1.0, 0.7, 0.4, ..., 0.23],
    "learning_rate": [0.001, 0.001, 0.001, 0.0005, ..., 0.000125]
  },
  
  "train_image_ids": [1, 3, 5, 7, ..., 199],
  "val_image_ids": [2, 4, 6, 8, ..., 200],
  "all_image_ids": [1, 2, 3, 4, ..., 200],
  
  "trained_at": "2026-01-07T14:30:15.123456"
}
```

## Key Information Categories

### 1. Basic Training Information
- `target_feature`: Feature being predicted
- `training_mode`: Classification or regression
- `num_classes`: Number of classes (classification only)
- `use_sigmoid`: Whether sigmoid activation was used

### 2. Training Configuration (`training_config`)
All hyperparameters used for training:
- Split ratio, epochs, learning rate, batch size
- Augmentation, normalization flags
- LR scheduling, differential LR settings
- **Regularization:** dropout, weight_decay
- **Early stopping:** patience, min_delta

### 3. Regularization (`regularization`)
- `dropout`: Dropout rate applied in model head
- `weight_decay`: L2 regularization strength

### 4. Early Stopping (`early_stopping`)
- `enabled`: Whether early stopping was active
- `patience`: Number of epochs to wait
- `triggered`: Whether it actually stopped training
- `stopped_epoch`: When it stopped (if triggered)
- `best_epoch`: Best epoch number
- `best_val_loss`: Best validation loss achieved
- `total_epochs_trained`: Actual number of epochs trained

### 5. Learning Rate Optimization
- **LR Scheduling** (`lr_scheduling`): ReduceLROnPlateau configuration and final LR
- **Differential LR** (`differential_lr`): Separate learning rates for backbone and head

### 6. Class Weights (`class_weights`)
- Classification only
- Automatic inverse frequency weighting
- Stores actual weights used in loss function

### 7. Normalization (`normalization`)
- Regression only
- Min-max normalization configuration
- Train set statistics

### 8. Augmentation (`augmentation`)
- Data augmentation configuration
- Transformation parameters
- Warping settings

### 9. Model Architecture (`model`)
- Model type, architecture, backbone
- **Dropout**: Actual dropout rate in model
- **use_sigmoid**: Output activation
- Parameter counts (total, trainable, frozen)
- Complete head layer structure with Dropout info
- Pre-trained weights source

### 10. Dataset Information (`dataset`, `split_quality`)
- Sample counts, batch counts
- Split strategy
- Class distribution (classification)
- Split quality warnings

### 11. Metrics (`train_metrics`, `val_metrics`)
- Loss, accuracy, F1-score
- Per-class metrics (precision, recall, F1)
- Confusion matrices

### 12. Training History (`training_history`)
- Epoch-by-epoch loss values
- **Learning rate changes** over time

### 13. Data Traceability
- `train_image_ids`: IDs of training images
- `val_image_ids`: IDs of validation images
- `all_image_ids`: All image IDs used
- `trained_at`: Timestamp

## Usage

### Loading Metadata
```python
import json

# Load from model file
metadata = torch.load('model.pth')['metadata']

# Or from separate JSON file
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)
```

### Accessing Key Information
```python
# Check if early stopping was triggered
if metadata['early_stopping']['triggered']:
    print(f"Stopped at epoch {metadata['early_stopping']['stopped_epoch']}")
    print(f"Best: epoch {metadata['early_stopping']['best_epoch']}, loss {metadata['early_stopping']['best_val_loss']}")

# Get regularization settings
dropout = metadata['regularization']['dropout']
weight_decay = metadata['regularization']['weight_decay']

# Get final learning rate
final_lr = metadata['lr_scheduling']['final_lr']

# Get class weights (classification)
if metadata['class_weights']['enabled']:
    weights = metadata['class_weights']['weights']
    print(f"Class weights: {weights}")

# Get model architecture details
dropout_from_model = metadata['model']['dropout']
total_params = metadata['model']['total_parameters']
```

## Frontend Display

All metadata is displayed in `ai_training_overview.html`:
- ✅ Regularization (Dropout, Weight Decay)
- ✅ Early Stopping (status, best epoch, triggered info)
- ✅ LR Scheduling (final LR)
- ✅ Differential LR (backbone/head LR)
- ✅ Class Weights (per-class weights)
- ✅ Complete model architecture
- ✅ Training history charts

## Version History

- **v1.1.5 (2026-01-07)**: Added regularization, early stopping, enhanced model info
- **v1.1.4 (2026-01-07)**: Added LR scheduling, differential LR
- **v1.1.3 (2026-01-07)**: Added class weights
- **v1.1.2 and earlier**: Basic training metadata

---

**Note:** All new features are backward compatible. Old models will load correctly with default values for missing fields.
