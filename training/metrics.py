"""Binary segmentation metrics for PyTorch tensors."""

from __future__ import annotations

import torch


def _ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure metrics always operate on a leading batch dimension."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor


def _prepare_predictions(preds: torch.Tensor, threshold: float) -> torch.Tensor:
    """Convert logits or probabilities into a binary prediction mask."""
    preds = _ensure_batch_dim(preds).float()

    # If values fall outside [0, 1], treat them as logits and apply sigmoid.
    if torch.any(preds < 0) or torch.any(preds > 1):
        preds = torch.sigmoid(preds)

    return (preds >= threshold).float()


def _prepare_targets(targets: torch.Tensor) -> torch.Tensor:
    """Convert target masks into binary float tensors."""
    targets = _ensure_batch_dim(targets).float()
    return (targets > 0.5).float()


def binary_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Compute mean binary IoU over a batch.

    `preds` may be raw logits or probabilities. If any prediction value is
    outside `[0, 1]`, the tensor is treated as logits and passed through
    `sigmoid` before thresholding.
    """
    pred_mask = _prepare_predictions(preds, threshold)
    target_mask = _prepare_targets(targets)

    intersection = (pred_mask * target_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)

    return float(iou.mean().item())


def binary_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Compute mean binary Dice score over a batch.

    `preds` may be raw logits or probabilities. If any prediction value is
    outside `[0, 1]`, the tensor is treated as logits and passed through
    `sigmoid` before thresholding.
    """
    pred_mask = _prepare_predictions(preds, threshold)
    target_mask = _prepare_targets(targets)

    intersection = (pred_mask * target_mask).sum(dim=(1, 2, 3))
    denominator = pred_mask.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)

    return float(dice.mean().item())
