"""
Training Visualization Module

Generates plots and visualizations for training analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Dict, List, Optional


def generate_loss_plot(
    train_loss: List[float],
    val_loss: List[float],
    title: str = "Training Loss History",
    figsize: tuple = (10, 6),
    dpi: int = 100
) -> io.BytesIO:
    """
    Generate loss plot showing training and validation loss over epochs.
    
    Args:
        train_loss: List of training loss values per epoch
        val_loss: List of validation loss values per epoch
        title: Plot title
        figsize: Figure size (width, height)
        dpi: DPI for image quality
    
    Returns:
        BytesIO buffer containing PNG image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    epochs = range(1, len(train_loss) + 1)
    
    # Plot lines
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add min val loss marker
    min_val_epoch = np.argmin(val_loss) + 1
    min_val_loss = min(val_loss)
    ax.axvline(x=min_val_epoch, color='green', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(min_val_epoch, min_val_loss, f'  Best: Epoch {min_val_epoch}', 
            fontsize=9, color='green', verticalalignment='bottom')
    
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)
    
    # Set x-axis to show all epochs
    ax.set_xlim(left=1, right=len(train_loss))
    ax.set_xticks(list(epochs))
    
    # Tight layout
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def generate_metrics_summary(
    train_metrics: Dict,
    val_metrics: Dict,
    training_mode: str = "regression"
) -> str:
    """
    Generate text summary of training metrics.
    
    Args:
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        training_mode: 'regression' or 'classification'
    
    Returns:
        Formatted text summary
    """
    if training_mode == "regression":
        summary = f"""
Training Performance Summary:
=============================

Training Set:
  R² Score: {train_metrics.get('r2_score', 0):.4f}
  RMSE: {train_metrics.get('rmse', 0):.4f}
  MAE: {train_metrics.get('mae', 0):.4f}
  Samples: {train_metrics.get('num_samples', 0)}

Validation Set:
  R² Score: {val_metrics.get('r2_score', 0):.4f}
  RMSE: {val_metrics.get('rmse', 0):.4f}
  MAE: {val_metrics.get('mae', 0):.4f}
  Samples: {val_metrics.get('num_samples', 0)}

Overfitting Check:
  R² Difference: {abs(train_metrics.get('r2_score', 0) - val_metrics.get('r2_score', 0)):.4f}
  RMSE Ratio: {val_metrics.get('rmse', 1) / max(train_metrics.get('rmse', 1), 0.0001):.2f}x
"""
    else:
        summary = f"""
Training Performance Summary:
=============================

Training Set:
  Accuracy: {train_metrics.get('accuracy', 0) * 100:.2f}%
  F1 Score: {train_metrics.get('f1_score_macro', 0) * 100:.2f}%
  Samples: {train_metrics.get('num_samples', 0)}

Validation Set:
  Accuracy: {val_metrics.get('accuracy', 0) * 100:.2f}%
  F1 Score: {val_metrics.get('f1_score_macro', 0) * 100:.2f}%
  Samples: {val_metrics.get('num_samples', 0)}

Overfitting Check:
  Accuracy Difference: {abs(train_metrics.get('accuracy', 0) - val_metrics.get('accuracy', 0)) * 100:.2f}%
"""
    
    return summary.strip()
