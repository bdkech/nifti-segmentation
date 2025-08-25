"""Weights & Biases integration for experiment tracking."""

import os
from pathlib import Path
from typing import Any, Optional

import wandb
from loguru import logger

from ..utils.config import WandbConfig


def setup_wandb(config: WandbConfig) -> None:
    """Initialize Weights & Biases with configuration.
    
    Args:
        config: W&B configuration parameters
        
    Notes:
        - Sets up W&B environment variables and login
        - Configures project settings and run parameters
        - Handles authentication and API key management
    """
    if not config.enabled:
        logger.info("W&B logging disabled")
        return
    
    try:
        # Setup W&B environment
        os.environ["WANDB_PROJECT"] = config.project
        
        if config.entity:
            os.environ["WANDB_ENTITY"] = config.entity
        
        # Initialize W&B
        wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.name,
            tags=config.tags,
            notes=config.notes,
            reinit=True,  # Allow multiple runs in same process
        )
        
        logger.info(f"W&B initialized for project: {config.project}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        logger.info("Continuing without W&B logging")


def log_model_artifacts(
    model_path: Path,
    model_name: str = "unetr_model",
    model_type: str = "model",
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log model artifacts to W&B.
    
    Args:
        model_path: Path to model checkpoint file
        model_name: Name for the model artifact
        model_type: Type of model artifact
        metadata: Optional metadata dictionary
        
    Notes:
        - Logs model checkpoints as W&B artifacts
        - Includes model metadata and performance metrics
        - Enables model versioning and tracking
    """
    if not wandb.run:
        logger.warning("W&B run not active, skipping model artifact logging")
        return
    
    try:
        # Create model artifact
        model_artifact = wandb.Artifact(
            name=model_name,
            type=model_type,
            metadata=metadata or {},
        )
        
        # Add model file
        model_artifact.add_file(str(model_path))
        
        # Log artifact
        wandb.log_artifact(model_artifact)
        
        logger.info(f"Model artifact logged: {model_name}")
        
    except Exception:
        logger.exception("Failed to log model artifact")


def log_data_artifacts(
    data_dir: Path,
    artifact_name: str = "training_data",
    artifact_type: str = "dataset",
    description: Optional[str] = None,
) -> None:
    """Log dataset as W&B artifact.
    
    Args:
        data_dir: Directory containing training data
        artifact_name: Name for the dataset artifact
        artifact_type: Type of dataset artifact
        description: Optional description of the dataset
        
    Notes:
        - Logs training/validation datasets for reproducibility
        - Enables dataset versioning and lineage tracking
        - Useful for experiment comparison and debugging
    """
    if not wandb.run:
        logger.warning("W&B run not active, skipping data artifact logging")
        return
    
    try:
        # Create dataset artifact
        data_artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description or f"Dataset from {data_dir}",
        )
        
        # Add directory (recursively)
        data_artifact.add_dir(str(data_dir))
        
        # Log artifact
        wandb.log_artifact(data_artifact)
        
        logger.info(f"Data artifact logged: {artifact_name}")
        
    except Exception:
        logger.exception("Failed to log data artifact")


def log_config_artifact(
    config_dict: dict[str, Any],
    artifact_name: str = "experiment_config",
) -> None:
    """Log experiment configuration as W&B artifact.
    
    Args:
        config_dict: Configuration dictionary to log
        artifact_name: Name for the configuration artifact
        
    Notes:
        - Logs complete experiment configuration
        - Enables exact reproduction of experiments
        - Tracks hyperparameter changes across runs
    """
    if not wandb.run:
        logger.warning("W&B run not active, skipping config artifact logging")
        return
    
    try:
        # Create config artifact
        config_artifact = wandb.Artifact(
            name=artifact_name,
            type="config",
            metadata=config_dict,
        )
        
        # Create temporary config file
        config_file = Path("/tmp/experiment_config.yaml")
        import yaml
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Add config file
        config_artifact.add_file(str(config_file))
        
        # Log artifact
        wandb.log_artifact(config_artifact)
        
        # Cleanup temp file
        config_file.unlink(missing_ok=True)
        
        logger.info(f"Config artifact logged: {artifact_name}")
        
    except Exception:
        logger.exception("Failed to log config artifact")


def log_predictions(
    predictions: dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log model predictions and visualizations to W&B.
    
    Args:
        predictions: Dictionary containing prediction results
        step: Optional step number for logging
        
    Notes:
        - Logs segmentation predictions as images/videos
        - Includes metrics and performance statistics
        - Enables visual inspection of model performance
    """
    if not wandb.run:
        logger.warning("W&B run not active, skipping predictions logging")
        return
    
    try:
        # TODO: Implement prediction visualization
        # This would include:
        # - Converting 3D volumes to 2D slices for visualization
        # - Creating overlay images of predictions on original volumes
        # - Logging dice scores and other metrics per sample
        # - Creating interactive visualizations for 3D data
        
        # For now, just log basic metrics
        if "metrics" in predictions:
            wandb.log(predictions["metrics"], step=step)
        
        logger.debug("Predictions logged to W&B")
        
    except Exception:
        logger.exception("Failed to log predictions")


def finish_wandb_run() -> None:
    """Finish current W&B run and cleanup.
    
    Notes:
        - Properly closes W&B run and uploads remaining data
        - Should be called at the end of training/evaluation
        - Handles cleanup and resource management
    """
    if wandb.run:
        try:
            wandb.finish()
            logger.info("W&B run finished successfully")
        except Exception:
            logger.exception("Error finishing W&B run")
    else:
        logger.debug("No active W&B run to finish")


class WandbCallback:
    """Custom callback for W&B integration with Lightning training.
    
    This callback provides additional W&B logging functionality beyond
    the standard WandbLogger, including:
    - Custom artifact logging
    - Prediction visualization
    - Model performance tracking
    """
    
    def __init__(
        self,
        log_model_artifacts: bool = True,
        log_predictions: bool = False,
        prediction_log_interval: int = 10,
    ) -> None:
        """Initialize W&B callback.
        
        Args:
            log_model_artifacts: Whether to log model checkpoints as artifacts
            log_predictions: Whether to log prediction visualizations
            prediction_log_interval: Interval for logging predictions (epochs)
        """
        self.log_model_artifacts = log_model_artifacts
        self.log_predictions = log_predictions
        self.prediction_log_interval = prediction_log_interval
    
    def on_train_start(self, config: dict[str, Any]) -> None:
        """Called at the start of training.
        
        Args:
            config: Training configuration dictionary
        """
        if self.log_model_artifacts:
            log_config_artifact(config)
    
    def on_validation_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float],
        predictions: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called at the end of each validation epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics dictionary
            predictions: Optional predictions for visualization
        """
        # Log additional metrics
        if metrics:
            wandb.log({"epoch": epoch, **metrics})
        
        # Log predictions if enabled
        if (
            self.log_predictions
            and predictions
            and epoch % self.prediction_log_interval == 0
        ):
            log_predictions(predictions, step=epoch)
    
    def on_train_end(self, checkpoint_path: Optional[Path] = None) -> None:
        """Called at the end of training.
        
        Args:
            checkpoint_path: Path to final model checkpoint
        """
        # Log final model artifact
        if self.log_model_artifacts and checkpoint_path:
            log_model_artifacts(
                checkpoint_path,
                model_name="final_model",
                metadata={"training_complete": True},
            )
        
        # Finish W&B run
        finish_wandb_run()