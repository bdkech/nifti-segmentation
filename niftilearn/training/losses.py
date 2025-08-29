"""Loss functions for medical image segmentation training using MONAI.

This module provides a factory interface for MONAI's optimized loss functions
designed for medical image segmentation tasks.
"""

from typing import Any

import torch.nn as nn
from loguru import logger
from monai.losses import DiceCELoss, DiceLoss, FocalLoss


def get_loss_function(loss_name: str, **kwargs: Any) -> nn.Module:
    """Factory function to create MONAI loss functions by name.

    This function provides a unified interface to create various MONAI loss
    functions commonly used for medical image segmentation.

    Args:
        loss_name: Name of the loss function ('dice', 'focal', 'dicece')
        **kwargs: Additional arguments passed to the MONAI loss function

    Returns:
        Configured MONAI loss function instance

    Raises:
        ValueError: If loss_name is not recognized
    """
    # Registry of available MONAI loss functions
    loss_registry: dict[str, type[nn.Module]] = {
        "dice": DiceLoss,
        "focal": FocalLoss,
        "dicece": DiceCELoss,
        "dice_ce": DiceCELoss,  # Alternative name
        "dicecross": DiceCELoss,  # Another alternative
    }

    loss_name_lower = loss_name.lower()
    if loss_name_lower not in loss_registry:
        available_losses = list(loss_registry.keys())
        raise ValueError(
            f"Unknown loss function '{loss_name}'. "
            f"Available options: {available_losses}"
        )

    loss_class = loss_registry[loss_name_lower]

    # Apply default parameters for common use cases
    if loss_name_lower == "dice":
        # MONAI DiceLoss defaults: include_background=True, to_onehot_y=False, sigmoid=False
        # Override for binary segmentation if not specified
        default_kwargs = {"include_background": True, "sigmoid": True}
        default_kwargs.update(kwargs)
        loss_instance = loss_class(**default_kwargs)
    elif loss_name_lower == "focal":
        # MONAI FocalLoss defaults: include_background=True, alpha=None, gamma=2.0
        default_kwargs = {"include_background": True, "gamma": 2.0}
        default_kwargs.update(kwargs)
        loss_instance = loss_class(**default_kwargs)
    elif loss_name_lower in ["dicece", "dice_ce", "dicecross"]:
        # MONAI DiceCELoss defaults: include_background=True, to_onehot_y=False, sigmoid=False
        default_kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
        }
        default_kwargs.update(kwargs)
        loss_instance = loss_class(**default_kwargs)
    else:
        loss_instance = loss_class(**kwargs)

    logger.info(
        f"Created MONAI {loss_class.__name__} with parameters: {kwargs}"
    )
    return loss_instance


def get_default_loss_config(loss_name: str) -> dict[str, Any]:
    """Get default configuration for a specific loss function.

    Args:
        loss_name: Name of the loss function

    Returns:
        Dictionary of default parameters for the loss function
    """
    defaults = {
        "dice": {
            "include_background": True,
            "sigmoid": True,
            "squared_pred": False,
            "jaccard": False,
        },
        "focal": {
            "include_background": True,
            "alpha": None,
            "gamma": 2.0,
            "weight": None,
        },
        "dicece": {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "lambda_dice": 1.0,
            "lambda_ce": 1.0,
        },
    }

    loss_name_lower = loss_name.lower()
    if loss_name_lower in ["dice_ce", "dicecross"]:
        loss_name_lower = "dicece"

    return defaults.get(loss_name_lower, {})
