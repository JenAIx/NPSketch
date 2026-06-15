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
from .warmup_scheduler import WarmupScheduler
from utils.logger import get_logger
from config import get_config

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True, min_epochs: int = 3):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in monitored value to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
            min_epochs: Minimum epochs before early stopping can trigger (default: 3)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs
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
            epoch: Current epoch number (0-based)
            
        Returns:
            True if training should stop, False otherwise
        """
        # Always track best loss and model state
        if self.best_loss is None or val_loss < (self.best_loss - self.min_delta):
            # First epoch or improvement found
            self.best_loss = val_loss
            self.best_epoch = epoch + 1  # Store as 1-based for consistency
            self.counter = 0
            if self.restore_best_weights:
                # Clone tensors to avoid in-place modifications during training
                # Memory: ~47 MB for ResNet-18 (only stored when best improves, not every epoch)
                self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch > 0:  # Don't log on first epoch
                logger.debug(f"Early stopping: Validation loss improved to {val_loss:.6f}")
            return False
        
        # No improvement - but don't trigger early stopping before min_epochs
        if epoch < self.min_epochs:
            logger.debug(f"Early stopping: Epoch {epoch+1} < min_epochs ({self.min_epochs}), continuing training")
            return False
        
        # No improvement and past min_epochs
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
        pos_weight: list = None,
        use_lr_scheduling: bool = True,
        use_differential_lr: bool = True,
        backbone_lr_multiplier: float = 0.1,
        dropout: float = 0.5,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.001,
        early_stopping_min_epochs: int = 3,
        lr_scheduling_patience: int = 5,
        lr_scheduling_threshold: float = 0.001,
        label_smoothing: float = 0.0,
        warmup_enabled: bool = False,
        warmup_epochs: int = 3
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
            early_stopping_min_epochs: Minimum epochs before early stopping can trigger (default: 3)
            lr_scheduling_patience: LR scheduler patience (default: 5)
            lr_scheduling_threshold: LR scheduler threshold (default: 0.001)
            label_smoothing: Label smoothing for classification (default: 0.0)
            warmup_enabled: Enable learning rate warmup (default: False)
            warmup_epochs: Number of warmup epochs (default: 3)
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
        self.lr_scheduling_patience = lr_scheduling_patience
        self.lr_scheduling_threshold = lr_scheduling_threshold
        self.label_smoothing = label_smoothing
        self.warmup_enabled = warmup_enabled
        self.warmup_epochs = warmup_epochs
        
        # Reproducibility: seed PyTorch (NumPy is seeded in the split strategy).
        # Without this, weight init / dropout / shuffling differ between runs.
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Auto-determine use_sigmoid if not specified.
        # Regression now uses a LINEAR output head: with min-max targets the bulk
        # of this dataset sits near 1.0, exactly in the sigmoid saturation zone
        # (vanishing gradients at score extremes). Predictions are clamped to the
        # valid range after denormalization instead.
        if use_sigmoid is None:
            use_sigmoid = False

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
                # Configure CPU threads from training config (default to 8)
                config = get_config()
                threads = config.get("performance.torch_num_threads", 8)
                try:
                    torch.set_num_threads(max(1, int(threads)))
                except (TypeError, ValueError):
                    torch.set_num_threads(8)
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
                self.criterion = nn.CrossEntropyLoss(
                    weight=weights_tensor,
                    label_smoothing=label_smoothing
                )
                logger.info(f"Loss function: CrossEntropyLoss with class weights (classification, {num_outputs} classes)")
                logger.info(f"  Class weights: {[f'{w:.4f}' for w in class_weights]}")
                if label_smoothing > 0:
                    logger.info(f"  Label smoothing: {label_smoothing}")
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                logger.info(f"Loss function: CrossEntropyLoss without weights (classification, {num_outputs} classes)")
                logger.warning("  No class weights provided - classes will be equally weighted")
                if label_smoothing > 0:
                    logger.info(f"  Label smoothing: {label_smoothing}")
        elif training_mode == "components":
            # Multi-label: 60 independent binary sub-labels (20 elements x PRES/ACC/POS).
            # BCEWithLogitsLoss applies the sigmoid internally (stable); the model
            # outputs raw logits (use_sigmoid=False).
            pw = None
            if pos_weight is not None:
                pw = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            logger.info(f"Loss function: BCEWithLogitsLoss (components, {num_outputs} sub-labels)")
            if pw is not None:
                logger.info(f"  pos_weight enabled (per-label, len {len(pos_weight)})")
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
                patience=lr_scheduling_patience,  # Task-specific patience
                verbose=False,         # We'll log manually
                min_lr=1e-6,          # Minimum LR
                threshold=lr_scheduling_threshold  # Task-specific threshold
            )
            logger.info(f"Learning Rate Scheduling enabled:")
            logger.info(f"  Strategy: ReduceLROnPlateau")
            logger.info(f"  Factor: 0.5 (halve LR)")
            logger.info(f"  Patience: {lr_scheduling_patience} epochs")
            logger.info(f"  Threshold: {lr_scheduling_threshold}")
            logger.info(f"  Min LR: 1e-6")
        
        # Warmup scheduler (for classification)
        self.warmup_scheduler = None
        if warmup_enabled and warmup_epochs > 0:
            self.warmup_scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                base_lr=learning_rate,
                warmup_start_lr=0.0
            )
            logger.info(f"Learning Rate Warmup enabled:")
            logger.info(f"  Warmup epochs: {warmup_epochs}")
            logger.info(f"  Start LR: 0.0 → Target LR: {learning_rate}")
        
        # Early stopping
        self.early_stopping = None
        if early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=True,
                min_epochs=early_stopping_min_epochs
            )
            logger.info(f"Early Stopping enabled:")
            logger.info(f"  Patience: {early_stopping_patience} epochs")
            logger.info(f"  Min delta: {early_stopping_min_delta}")
            logger.info(f"  Min epochs: {early_stopping_min_epochs}")
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
        callback: Optional[Callable] = None,
        epoch: int = 0
    ) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            callback: Optional callback function(epoch, batch, loss)
            epoch: Current epoch number (for warmup scheduler)
        
        Returns:
            Metrics dict with train_loss, val_loss
        """
        # Apply warmup if enabled and in warmup phase
        if self.warmup_scheduler and self.warmup_scheduler.is_warming_up(epoch):
            warmup_lr = self.warmup_scheduler.step(epoch)
            logger.info(f"Warmup epoch {epoch+1}/{self.warmup_epochs}: LR = {warmup_lr:.6f}")
        
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
        
        # Learning rate scheduling (only after warmup is complete)
        if self.scheduler is not None and val_loss is not None:
            # Skip LR scheduling during warmup phase
            if not (self.warmup_scheduler and self.warmup_scheduler.is_warming_up(epoch)):
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

        # Components mode: collect 60-dim probabilities + targets (keep 2D)
        if self.training_mode == "components":
            probs_list, tgt_list = [], []
            with torch.no_grad():
                for images, targets in data_loader:
                    images = images.to(self.device)
                    outputs = self.model(images)              # logits (B, 60)
                    probs = torch.sigmoid(outputs)
                    probs_list.append(probs.cpu().numpy())
                    tgt_list.append(targets.cpu().numpy())
            probs = np.concatenate(probs_list, axis=0)
            tgts = np.concatenate(tgt_list, axis=0)
            return self._calculate_component_metrics(probs, tgts)

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
                    # Regression: get raw output values.
                    # No pre-denormalization clamping here - predictions are
                    # clamped to the valid score range AFTER denormalization
                    # in _calculate_regression_metrics (matches inference).
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
            predictions = self.normalizer.inverse_transform(predictions)
            targets = self.normalizer.inverse_transform(targets)

            # Clamp predictions to the valid score range (linear head can
            # overshoot slightly; matches inference behavior)
            if self.normalizer.min_value is not None and self.normalizer.max_value is not None:
                predictions = np.clip(predictions, self.normalizer.min_value, self.normalizer.max_value)

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

        # Per-score-bin metrics (decades on the denormalized scale): makes
        # performance at the score extremes visible, which the global R²/MAE
        # hide on a heavily skewed distribution.
        per_score_bin = {}
        bin_width = 10.0
        bin_starts = np.floor(targets / bin_width) * bin_width
        for start in sorted(np.unique(bin_starts)):
            bin_mask = bin_starts == start
            bin_t = targets[bin_mask]
            bin_p = predictions[bin_mask]
            label = f"{int(start)}-{int(start + bin_width - 1)}"
            per_score_bin[label] = {
                'count': int(bin_mask.sum()),
                'mae': float(np.mean(np.abs(bin_p - bin_t))),
                'rmse': float(np.sqrt(np.mean((bin_p - bin_t) ** 2))),
                'mean_pred': float(np.mean(bin_p)),
                'mean_target': float(np.mean(bin_t))
            }

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'per_score_bin': per_score_bin,
            'predictions': predictions.tolist()[:1000],
            'targets': targets.tolist()[:1000],
            'num_samples': len(targets)
        }

    def _calculate_component_metrics(self, probs: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Metrics for the 60-sub-label component model.

        probs, targets: (N, 60), column order ELEM01PRES, ELEM01ACC, ELEM01POS,
        ELEM02PRES, ... (3 aspects interleaved per element).

        Reports: per-sub-label macro F1/accuracy, per-aspect (presence/accuracy/
        position), per-component (20), AND the derived Total_Score (= sum of the 60)
        as R²/RMSE/MAE + per-score-bin — the direct comparison to the holistic model.
        """
        preds = (probs >= 0.5).astype(int)
        tgt = targets.astype(int)

        def prf(p, t):
            tp = int(((p == 1) & (t == 1)).sum())
            fp = int(((p == 1) & (t == 0)).sum())
            fn = int(((p == 0) & (t == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            acc = float((p == t).mean())
            return prec, rec, f1, acc

        # per sub-label (60), then macro
        per_label_f1 = []
        for j in range(probs.shape[1]):
            _, _, f1, _ = prf(preds[:, j], tgt[:, j])
            per_label_f1.append(f1)
        macro_f1 = float(np.mean(per_label_f1))
        overall_acc = float((preds == tgt).mean())

        # per aspect: columns j%3 == 0 presence, 1 accuracy, 2 position
        aspect = {}
        for name, off in (("presence", 0), ("accuracy", 1), ("position", 2)):
            cols = list(range(off, probs.shape[1], 3))
            _, _, f1, acc = prf(preds[:, cols], tgt[:, cols])
            aspect[name] = {"f1": float(f1), "accuracy": float(acc)}

        # per component (20): each element = its 3 columns
        per_component = {}
        for e in range(20):
            cols = [3 * e, 3 * e + 1, 3 * e + 2]
            _, _, f1, acc = prf(preds[:, cols], tgt[:, cols])
            per_component[f"E{e+1:02d}"] = {"f1": float(f1), "accuracy": float(acc)}

        # derived Total_Score (hard = sum of >0.5 decisions; soft = sum of probs)
        score_true = tgt.sum(axis=1).astype(float)
        score_hard = preds.sum(axis=1).astype(float)
        score_soft = probs.sum(axis=1)

        def score_stats(pred_score):
            mse = float(np.mean((pred_score - score_true) ** 2))
            ss_res = np.sum((score_true - pred_score) ** 2)
            ss_tot = np.sum((score_true - score_true.mean()) ** 2)
            return {
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(pred_score - score_true))),
                "r2_score": float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0,
            }

        # per-score-bin on the hard derived score (decades) — comparable to holistic
        per_score_bin = {}
        bin_starts = np.floor(score_true / 10.0) * 10.0
        for start in sorted(np.unique(bin_starts)):
            m = bin_starts == start
            per_score_bin[f"{int(start)}-{int(start+9)}"] = {
                "count": int(m.sum()),
                "mae": float(np.mean(np.abs(score_hard[m] - score_true[m]))),
                "rmse": float(np.sqrt(np.mean((score_hard[m] - score_true[m]) ** 2))),
                "mean_pred": float(np.mean(score_hard[m])),
                "mean_target": float(np.mean(score_true[m])),
            }

        # For frontend/early-stopping compatibility, surface the derived-score
        # regression metrics at the top level (val_loss is BCE; these are the
        # interpretable headline numbers).
        hard = score_stats(score_hard)
        return {
            "macro_f1": macro_f1,
            "sublabel_accuracy": overall_acc,
            "per_aspect": aspect,
            "per_component": per_component,
            "derived_score_hard": hard,
            "derived_score_soft": score_stats(score_soft),
            "r2_score": hard["r2_score"],
            "rmse": hard["rmse"],
            "mae": hard["mae"],
            "per_score_bin": per_score_bin,
            "num_samples": int(len(score_true)),
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
                callback=lambda batch, loss: callback(epoch, batch, loss) if callback else None,
                epoch=epoch
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