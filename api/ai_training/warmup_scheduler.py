"""
Learning Rate Warmup Scheduler for CNN Training

Provides linear learning rate warmup for the first few epochs to stabilize training.
"""

import torch


class WarmupScheduler:
    """Learning rate warmup scheduler with linear warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, base_lr: float, warmup_start_lr: float = 0.0):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            base_lr: Target learning rate after warmup
            warmup_start_lr: Starting learning rate (default: 0.0)
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0
        
        # Store initial LRs for differential learning rates
        self.initial_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def step(self, epoch: int):
        """
        Update learning rate based on current epoch.
        
        Args:
            epoch: Current epoch number (0-based)
        
        Returns:
            Primary learning rate (head LR for differential LR)
        """
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup: gradually increase from warmup_start_lr to base_lr
            warmup_factor = (epoch + 1) / self.warmup_epochs
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                # Scale each param group proportionally
                target_lr = self.initial_lrs[i]
                param_group['lr'] = self.warmup_start_lr + (target_lr - self.warmup_start_lr) * warmup_factor
            
            # Return primary LR (last param group = head for differential LR)
            return self.optimizer.param_groups[-1]['lr']
        
        # After warmup, return base LR (no change)
        return self.base_lr
    
    def is_warming_up(self, epoch: int) -> bool:
        """
        Check if still in warmup phase.
        
        Args:
            epoch: Current epoch number (0-based)
        
        Returns:
            True if in warmup phase, False otherwise
        """
        return epoch < self.warmup_epochs
