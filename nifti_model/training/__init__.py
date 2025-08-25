"""Training and experiment tracking submodule.

This submodule provides:
- Lightning training manager with callbacks
- Weights & Biases integration for experiment tracking
- Model checkpointing and validation workflows
"""

from .logging import (
    WandbCallback,
    finish_wandb_run,
    log_config_artifact,
    log_data_artifacts,
    log_model_artifacts,
    log_predictions,
    setup_wandb,
)
from .trainer import TrainingManager, create_trainer_from_config

__all__ = [
    # Training
    "TrainingManager",
    "WandbCallback",
    "create_trainer_from_config",
    "finish_wandb_run",
    "log_config_artifact",
    "log_data_artifacts",
    "log_model_artifacts",
    "log_predictions",
    # Experiment logging
    "setup_wandb",
]