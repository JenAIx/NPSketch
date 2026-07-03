"""
Custom loss functions for the training pipeline.

AsymmetricLoss (ASL) — Ridnik et al., "Asymmetric Loss for Multi-Label
Classification", ICCV 2021 (arXiv:2009.14119). A drop-in replacement for
BCEWithLogitsLoss on the 60-sub-label component head. It decouples the focusing
of positives vs negatives and hard-thresholds (clips) very-easy negatives, which
targets the strong positive/negative imbalance across the sparse sub-labels
better than a per-label pos_weight can. Operates on raw logits (the model head
stays linear / use_sigmoid=False), same as BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Multi-label asymmetric loss on logits.

    Args:
        gamma_neg: focusing parameter for negative samples (higher = down-weight
            easy negatives more). Default 4.
        gamma_pos: focusing parameter for positive samples. Default 1 (positives
            are rare/valuable → focus them less than negatives).
        clip: probability shift for negatives — an easy negative with
            (1 - p) > 1 - clip contributes zero loss (asymmetric hard-thresholding).
            Default 0.05. Set 0 to disable.
        eps: log stability.
        reduction: 'mean' (per-element, comparable to BCEWithLogitsLoss) or 'sum'.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits, targets: [B, num_labels]; targets in {0, 1} (float)."""
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric clipping: treat very-easy negatives as fully solved.
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)          # 1 - pt for negatives (post-clip)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        loss = -loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
