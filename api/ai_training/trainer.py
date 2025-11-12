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
from .dataset import create_dataloaders


class CNNTrainer:
    """CNN Trainer for drawing assessment using PyTorch."""
    
    def __init__(
        self,
        num_outputs: int = 1,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize CNN trainer.
        
        Args:
            num_outputs: Number of output features to predict
            learning_rate: Learning rate for optimizer
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = DrawingClassifier(num_outputs=num_outputs, pretrained=True)
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
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
                
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
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
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
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
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        
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
            
            # Print progress
            if metrics['val_loss'] is not None:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {metrics['train_loss']:.4f}, "
                      f"Val Loss = {metrics['val_loss']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: "
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
        
        print(f"Model saved: {filepath}")
        
        # Save metadata JSON
        if metadata:
            import json
            metadata['model_filename'] = filepath.name
            metadata['saved_at'] = timestamp
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved: {metadata_filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Model loaded: {filepath}")
    
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
