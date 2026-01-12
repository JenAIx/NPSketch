"""
CNN Trainer for Drawing Assessment (PyTorch)

Trains CNN models to predict features/scores from neuropsychological drawings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Callable, Optional
from pathlib import Path

from .model import DrawingClassifier
from .dataset import create_dataloaders, create_augmented_dataloaders
from utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in monitored value to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if best
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            # First epoch (epoch is 0-based, store as 1-based for consistency)
            self.best_loss = val_loss
            self.best_epoch = epoch + 1  # Store as 1-based for consistency with stopped_epoch
            if self.restore_best_weights:
                # Clone tensors to avoid in-place modifications during training
                # Memory: ~47 MB for ResNet-18 (only stored when best improves, not every epoch)
                self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        
        # Check if loss improved
        if val_loss < (self.best_loss - self.min_delta):
            # Improvement (epoch is 0-based, store as 1-based for consistency)
            self.best_loss = val_loss
            self.best_epoch = epoch + 1  # Store as 1-based for consistency with stopped_epoch
            self.counter = 0
            if self.restore_best_weights:
                # Clone tensors to avoid in-place modifications during training
                # Memory: ~47 MB for ResNet-18 (only stored when best improves, not every epoch)
                # Old best_model_state is automatically garbage collected
                self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            logger.debug(f"Early stopping: Validation loss improved to {val_loss:.6f}")
            return False
        else:
            # No improvement
            self.counter += 1
            logger.debug(f"Early stopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                logger.info(f"Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch} (1-based)")
                return True
            
            return False
    
    def restore_best(self, model: nn.Module):
        """Restore best model weights."""
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model from epoch {self.best_epoch}")
        else:
            logger.warning("No best model state available to restore")


class CNNTrainer:
    """CNN Trainer for drawing assessment using PyTorch."""
    
    def __init__(
        self,
        num_outputs: int = 1,
        learning_rate: float = 0.001,
        device: str = None,
        normalizer=None,
        training_mode: str = "regression",
        use_sigmoid: bool = None,
        class_weights: list = None,
        use_lr_scheduling: bool = True,
        use_differential_lr: bool = True,
        backbone_lr_multiplier: float = 0.1,
        dropout: float = 0.5,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.001
    ):
        """
        Initialize CNN trainer.
        
        Args:
            num_outputs: Number of output features to predict
            learning_rate: Learning rate for optimizer (head LR if differential LR is used)
            device: 'cuda', 'cpu', or None (auto-detect)
            normalizer: Target normalizer (None for classification)
            training_mode: 'regression' or 'classification'
            use_sigmoid: Use Sigmoid at output (auto: True for regression with normalizer, False otherwise)
            class_weights: Optional class weights for CrossEntropyLoss (classification only)
            use_lr_scheduling: Enable ReduceLROnPlateau scheduling (default: True)
            use_differential_lr: Use different LRs for backbone vs head (default: True)
            backbone_lr_multiplier: Backbone LR multiplier (default: 0.1 = 10x smaller)
            dropout: Dropout rate (default: 0.5)
            weight_decay: L2 regularization strength (default: 0.0001)
            early_stopping_patience: Early stopping patience (0 = disabled, default: 15)
            early_stopping_min_delta: Minimum delta for early stopping (default: 0.001)
        """
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.normalizer = normalizer
        self.training_mode = training_mode
        self.class_weights = class_weights
        self.use_lr_scheduling = use_lr_scheduling
        self.use_differential_lr = use_differential_lr
        self.backbone_lr_multiplier = backbone_lr_multiplier
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        # Auto-determine use_sigmoid if not specified
        if use_sigmoid is None:
            # Use Sigmoid for regression with normalization
            use_sigmoid = (training_mode == "regression" and normalizer is not None)
        
        self.use_sigmoid = use_sigmoid
        
        # Auto-detect device
        if device is None:
            # Check for CUDA first, then MPS (Apple Silicon), then CPU
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
                # Optimize CPU performance on M1
                torch.set_num_threads(8)  # M1 has 8 performance cores
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model with dropout
        self.model = DrawingClassifier(
            num_outputs=num_outputs, 
            pretrained=True, 
            use_sigmoid=use_sigmoid,
            dropout=dropout
        )
        self.model.to(self.device)
        
        if use_sigmoid:
            logger.info(f"Using Sigmoid output activation (output range: [0, 1])")
        
        logger.info(f"Dropout rate: {dropout}")
        
        # Optimizer with optional differential learning rates
        if use_differential_lr:
            # Separate parameter groups: Backbone (smaller LR) and Head (normal LR)
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'fc' in name:  # Head layers (fc = final fully connected)
                    head_params.append(param)
                else:  # Backbone layers (conv1, bn1, layer1-4)
                    backbone_params.append(param)
            
            backbone_lr = learning_rate * backbone_lr_multiplier
            head_lr = learning_rate
            
            self.optimizer = optim.Adam([
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
            ])
            
            logger.info(f"Differential Learning Rates enabled:")
            logger.info(f"  Backbone LR: {backbone_lr:.6f} ({len(backbone_params)} param groups)")
            logger.info(f"  Head LR: {head_lr:.6f} ({len(head_params)} param groups)")
        else:
            # Standard: All parameters with same LR
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            logger.info(f"Standard Learning Rate: {learning_rate:.6f} (all parameters)")
        
        if weight_decay > 0:
            logger.info(f"Weight decay (L2 regularization): {weight_decay}")
        
        if training_mode == "classification":
            # Use class weights if provided
            if class_weights is not None:
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
                logger.info(f"Loss function: CrossEntropyLoss with class weights (classification, {num_outputs} classes)")
                logger.info(f"  Class weights: {[f'{w:.4f}' for w in class_weights]}")
            else:
                self.criterion = nn.CrossEntropyLoss()
                logger.info(f"Loss function: CrossEntropyLoss without weights (classification, {num_outputs} classes)")
                logger.warning("  No class weights provided - classes will be equally weighted")
        else:
            self.criterion = nn.MSELoss()
            logger.info(f"Loss function: MSELoss (regression)")
        
        # Learning rate scheduler
        self.scheduler = None
        if use_lr_scheduling:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',           # Minimize validation loss
                factor=0.5,           # Reduce LR by 50%
                patience=5,            # Wait 5 epochs without improvement
                verbose=False,         # We'll log manually
                min_lr=1e-6,          # Minimum LR
                threshold=0.001       # Minimum change to count as improvement
            )
            logger.info(f"Learning Rate Scheduling enabled:")
            logger.info(f"  Strategy: ReduceLROnPlateau")
            logger.info(f"  Factor: 0.5 (halve LR)")
            logger.info(f"  Patience: 5 epochs")
            logger.info(f"  Min LR: 1e-6")
        
        # Early stopping
        self.early_stopping = None
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=True
            )
            logger.info(f"Early Stopping enabled:")
            logger.info(f"  Patience: {early_stopping_patience} epochs")
            logger.info(f"  Min delta: {early_stopping_min_delta}")
            logger.info(f"  Restore best weights: True")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],  # Track LR changes
            'epoch': []
        }
        
        # Model directory
        self.model_dir = Path("/app/data/models")
        self.model_dir.mkdir(exist_ok=True)
    
    def get_primary_learning_rate(self) -> float:
        """
        Get the primary learning rate (Head LR if differential LR is enabled, otherwise the main LR).
        This is the learning rate that should be tracked and reported in metadata.
        
        Returns:
            Primary learning rate (Head LR for differential LR, or main LR otherwise)
        """
        if self.use_differential_lr and len(self.optimizer.param_groups) > 1:
            # param_groups[1] is the head (primary LR)
            return self.optimizer.param_groups[1]['lr']
        else:
            # Single param group or no differential LR
            return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            callback: Optional callback function(epoch, batch, loss)
        
        Returns:
            Metrics dict with train_loss, val_loss
        """
        self.model.train()
        train_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            
            # Callback
            if callback:
                callback(batch_idx, loss.item())
        
        # Validation
        val_loss = None
        if val_loader:
            val_loss = self.evaluate(val_loader)
        
        # Learning rate scheduling (if enabled and validation loss available)
        if self.scheduler is not None and val_loss is not None:
            old_lr = self.get_primary_learning_rate()
            self.scheduler.step(val_loss)
            new_lr = self.get_primary_learning_rate()
            
            # Log LR changes (track primary LR, which is the Head LR for differential LR)
            if new_lr != old_lr:
                logger.info(f"Learning Rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        metrics = {
            'train_loss': np.mean(train_losses),
            'val_loss': val_loss,
            'learning_rate': self.get_primary_learning_rate()  # Track primary LR (Head LR for differential LR)
        }
        
        return metrics
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate model on validation/test data.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Average loss
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def evaluate_metrics(self, data_loader: DataLoader) -> Dict:
        """
        Evaluate model with comprehensive metrics.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Dict with MSE, RMSE, MAE, R², predictions
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                
                if self.training_mode == "classification":
                    # Get predicted class (argmax)
                    predicted_classes = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predicted_classes.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    # Regression: get raw output values
                    # Clamp to [0, 1] if using sigmoid (safety check)
                    if self.use_sigmoid:
                        outputs = torch.clamp(outputs, 0.0, 1.0)
                    all_predictions.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Debug: Log prediction range before denormalization (only if debug level)
        if self.training_mode == "regression" and len(predictions) > 0:
            logger.debug(f"Model outputs (before denormalization): min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
            logger.debug(f"use_sigmoid: {self.use_sigmoid}, has_normalizer: {self.normalizer is not None}")
        elif self.training_mode == "regression" and len(predictions) == 0:
            logger.warning("Empty predictions array in regression mode - skipping debug logging")
        
        if self.training_mode == "classification":
            # Classification metrics
            return self._calculate_classification_metrics(predictions, targets)
        else:
            # Regression metrics
            return self._calculate_regression_metrics(predictions, targets)
    
    def _calculate_regression_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate regression metrics (MAE, RMSE, R², etc.)"""
        # Denormalize if normalizer is provided
        if self.normalizer is not None:
            predictions_before = predictions.copy()
            targets_before = targets.copy()
            
            predictions = self.normalizer.inverse_transform(predictions)
            targets = self.normalizer.inverse_transform(targets)
            
            # Debug: Log after denormalization (only if debug level)
            logger.debug(f"After denormalization: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE
        mask = targets != 0
        if np.any(mask):
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = 0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'predictions': predictions.tolist()[:1000],
            'targets': targets.tolist()[:1000],
            'num_samples': len(targets)
        }
    
    def _calculate_classification_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate classification metrics (Accuracy, F1, Precision, Recall)"""
        # Accuracy
        accuracy = np.mean(predictions == targets)
        
        # Per-class metrics
        num_classes = self.num_outputs
        per_class_metrics = {}
        
        for class_id in range(num_classes):
            tp = np.sum((predictions == class_id) & (targets == class_id))
            fp = np.sum((predictions == class_id) & (targets != class_id))
            fn = np.sum((predictions != class_id) & (targets == class_id))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[f'class_{class_id}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(targets == class_id))
            }
        
        # Macro F1
        f1_scores = [m['f1'] for m in per_class_metrics.values()]
        macro_f1 = float(np.mean(f1_scores))
        
        # Confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_class, pred_class in zip(targets, predictions):
            confusion_matrix[int(true_class), int(pred_class)] += 1
        
        return {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),  # Keep for backward compatibility
            'f1_score_macro': float(macro_f1),  # Frontend expects this key
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix.tolist(),
            'predictions': predictions.tolist()[:1000],
            'targets': targets.tolist()[:1000],
            'num_samples': len(targets)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            callback: Callback function(epoch, metrics)
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # Train epoch
            metrics = self.train_epoch(
                train_loader,
                val_loader,
                callback=lambda batch, loss: callback(epoch, batch, loss) if callback else None
            )
            
            # Store history (learning_rate managed by caller to avoid duplication)
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(metrics['train_loss'])
            if metrics['val_loss'] is not None:
                self.history['val_loss'].append(metrics['val_loss'])
            # Note: learning_rate append removed to avoid duplication with training loop
            # Callers should append metrics.get('learning_rate', self.learning_rate) themselves
            
            # Log progress
            if metrics['val_loss'] is not None:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {metrics['train_loss']:.4f}, "
                      f"Val Loss = {metrics['val_loss']:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {metrics['train_loss']:.4f}")
            
            # Callback
            if callback:
                callback(epoch, None, metrics)
        
        return self.history
    
    def save_model(self, name: str = "model", metadata: Dict = None):
        """
        Save model weights and metadata.
        
        Args:
            name: Model name
            metadata: Additional metadata to save (training config, data info, etc.)
        
        Returns:
            Model filepath
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.model_dir / f"{name}_{timestamp}.pth"
        metadata_filepath = self.model_dir / f"{name}_{timestamp}_metadata.json"
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'num_outputs': self.num_outputs
        }, filepath)
        
        logger.info(f"Model saved: {filepath}")
        
        # Save metadata JSON
        if metadata:
            import json
            metadata['model_filename'] = filepath.name
            metadata['saved_at'] = timestamp
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved: {metadata_filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str):
        """
        Load model weights.
        
        Handles backward compatibility: Old models (without differential LR) have 1 param_group,
        new models (with differential LR) have 2 param_groups. If mismatch occurs, optimizer
        state is skipped (only needed for training, not for evaluation).
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Try to load optimizer state, but handle backward compatibility
        if 'optimizer_state_dict' in checkpoint:
            saved_param_groups = len(checkpoint['optimizer_state_dict']['param_groups'])
            current_param_groups = len(self.optimizer.param_groups)
            
            if saved_param_groups == current_param_groups:
                # Compatible: Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(f"Model and optimizer loaded: {filepath}")
            else:
                # Incompatible: Skip optimizer state (only needed for training, not evaluation)
                logger.warning(
                    f"Optimizer state mismatch: Saved model has {saved_param_groups} param_group(s), "
                    f"current optimizer has {current_param_groups}. Skipping optimizer state. "
                    f"This is normal for old models (pre-differential LR). Model weights loaded successfully."
                )
        else:
            logger.warning("No optimizer state found in checkpoint. Model weights loaded successfully.")
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Model loaded: {filepath}")
    
    def predict(self, image_tensor: torch.Tensor) -> float:
        """
        Predict feature value for a single image.
        
        Args:
            image_tensor: Image tensor (1, 274, 568)
        
        Returns:
            Predicted value
        """
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dim
            output = self.model(image_tensor)
            return output.item()