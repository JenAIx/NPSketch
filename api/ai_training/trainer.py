"""
CNN Trainer for Drawing Assessment (PyTorch)

Trains CNN models to predict features/scores from neuropsychological drawings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
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


class CNNTrainer:
    """CNN Trainer for drawing assessment using PyTorch."""
    
    def __init__(
        self,
        num_outputs: int = 1,
        learning_rate: float = 0.001,
        device: str = None,
        normalizer=None,
        training_mode: str = "regression"
    ):
        """
        Initialize CNN trainer.
        
        Args:
            num_outputs: Number of output features to predict
            learning_rate: Learning rate for optimizer
            device: 'cuda', 'cpu', or None (auto-detect)
            normalizer: Target normalizer (None for classification)
            training_mode: 'regression' or 'classification'
        """
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.normalizer = normalizer
        self.training_mode = training_mode
        
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
        
        # Initialize model
        self.model = DrawingClassifier(num_outputs=num_outputs, pretrained=True)
        self.model.to(self.device)
        
        # Optimizer and loss - conditional based on training mode
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if training_mode == "classification":
            self.criterion = nn.CrossEntropyLoss()
            logger.info(f"Loss function: CrossEntropyLoss (classification, {num_outputs} classes)")
        else:
            self.criterion = nn.MSELoss()
            logger.info(f"Loss function: MSELoss (regression)")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        # Model directory
        self.model_dir = Path("/app/data/models")
        self.model_dir.mkdir(exist_ok=True)
    
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
        
        metrics = {
            'train_loss': np.mean(train_losses),
            'val_loss': val_loss
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
                    all_predictions.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
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
            'macro_f1': float(macro_f1),
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
            
            # Store history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(metrics['train_loss'])
            if metrics['val_loss'] is not None:
                self.history['val_loss'].append(metrics['val_loss'])
            
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
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
