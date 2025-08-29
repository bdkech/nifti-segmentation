"""Metrics for evaluating medical image segmentation performance using MONAI.

This module provides a factory interface for MONAI's optimized metrics
designed for medical image segmentation evaluation.
"""

from typing import Any, Union

import torch
from loguru import logger
from monai.metrics import DiceMetric, MeanIoU


class SegmentationMetrics:
    """Combined segmentation metrics using MONAI's implementations.

    This class provides a unified interface for multiple MONAI metrics
    commonly used in segmentation evaluation.
    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: str = "mean",
        get_not_nans: bool = False,
    ) -> None:
        """Initialize segmentation metrics.

        Args:
            include_background: Whether to include background class in calculations
            reduction: Reduction method ('mean', 'sum', 'mean_batch', 'sum_batch', 'none')
            get_not_nans: Whether to return only non-NaN values
        """
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans

        # Initialize MONAI metrics
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

        self.iou_metric = MeanIoU(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans,
        )

        logger.debug(
            f"SegmentationMetrics initialized with include_background={include_background}, "
            f"reduction={reduction}, get_not_nans={get_not_nans}"
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self.dice_metric.reset()
        self.iou_metric.reset()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Update all metrics with batch predictions and targets.

        Args:
            y_pred: Predictions tensor [N, C, H, W] or [N, H, W]
            y: Ground truth tensor [N, C, H, W] or [N, H, W]
        """
        # Ensure predictions are probabilities (apply sigmoid if needed)
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)

        # MONAI metrics expect [N, C, H, W] format
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
            y = y.unsqueeze(1)

        self.dice_metric(y_pred, y)
        self.iou_metric(y_pred, y)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute all accumulated metrics.

        Returns:
            Dictionary containing computed metrics
        """
        dice_score = self.dice_metric.aggregate()
        iou_score = self.iou_metric.aggregate()

        return {
            "dice": dice_score,
            "iou": iou_score,
        }

    def compute_batch(
        self, y_pred: torch.Tensor, y: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute metrics for a single batch without accumulation.

        Args:
            y_pred: Predictions tensor [N, C, H, W] or [N, H, W]
            y: Ground truth tensor [N, C, H, W] or [N, H, W]

        Returns:
            Dictionary containing batch metrics
        """
        # Ensure predictions are probabilities (apply sigmoid if needed)
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)

        # MONAI metrics expect [N, C, H, W] format
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
            y = y.unsqueeze(1)

        # Create temporary metric instances for batch computation
        temp_dice = DiceMetric(
            include_background=self.include_background,
            reduction="none",  # Get per-sample results
            get_not_nans=self.get_not_nans,
        )

        temp_iou = MeanIoU(
            include_background=self.include_background,
            reduction="none",  # Get per-sample results
            get_not_nans=self.get_not_nans,
        )

        # Compute batch metrics
        dice_batch = temp_dice(y_pred, y)
        iou_batch = temp_iou(y_pred, y)

        return {
            "dice": dice_batch,
            "iou": iou_batch,
        }


def get_metric(metric_name: str, **kwargs: Any) -> Union[DiceMetric, MeanIoU]:
    """Factory function to create MONAI metrics by name.

    Args:
        metric_name: Name of the metric ('dice', 'iou')
        **kwargs: Additional arguments passed to the MONAI metric

    Returns:
        Configured MONAI metric instance

    Raises:
        ValueError: If metric_name is not recognized
    """
    metric_registry: dict[str, type] = {
        "dice": DiceMetric,
        "iou": MeanIoU,
        "mean_iou": MeanIoU,  # Alternative name
        "jaccard": MeanIoU,  # IoU is also known as Jaccard index
    }

    metric_name_lower = metric_name.lower()
    if metric_name_lower not in metric_registry:
        available_metrics = list(metric_registry.keys())
        raise ValueError(
            f"Unknown metric '{metric_name}'. "
            f"Available options: {available_metrics}"
        )

    metric_class = metric_registry[metric_name_lower]

    # Apply default parameters for binary segmentation
    default_kwargs = {
        "include_background": True,
        "reduction": "mean",
        "get_not_nans": False,
    }
    default_kwargs.update(kwargs)

    metric_instance = metric_class(**default_kwargs)

    logger.info(
        f"Created MONAI {metric_class.__name__} with parameters: {kwargs}"
    )
    return metric_instance


def get_default_metric_config(metric_name: str) -> dict[str, Any]:
    """Get default configuration for a specific metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Dictionary of default parameters for the metric
    """
    defaults = {
        "dice": {
            "include_background": True,
            "reduction": "mean",
            "get_not_nans": False,
            "ignore_empty": True,
        },
        "iou": {
            "include_background": True,
            "reduction": "mean",
            "get_not_nans": False,
            "ignore_empty": True,
        },
    }

    metric_name_lower = metric_name.lower()
    if metric_name_lower in ["mean_iou", "jaccard"]:
        metric_name_lower = "iou"

    return defaults.get(metric_name_lower, {})
