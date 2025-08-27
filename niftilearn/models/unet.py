"""UNet2D model implementation using MONAI for medical image segmentation."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from monai.networks.nets import UNet

from niftilearn.config.models import Config, ModelConfig


class UNet2D(pl.LightningModule):
    """2D UNet model for slice-based medical image segmentation.
    
    This Lightning module wraps MONAI's UNet architecture for 2D slice processing,
    designed to work with the volume-based batch processing pipeline where each
    batch contains all slices from a single volume.
    
    The model expects input tensors of shape [N, C, H, W] where:
    - N: Number of slices in the volume
    - C: Number of input channels (typically 1 for grayscale medical images)
    - H, W: Height and width of each slice
    
    Output tensors have the same shape [N, C, H, W] for segmentation masks.
    """

    def __init__(self, config: ModelConfig, training_config: dict = None) -> None:
        """Initialize UNet2D model.
        
        Args:
            config: Model configuration containing architecture parameters
            training_config: Training configuration dict (optional, for compatibility)
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        super().__init__()
        
        self.config = config
        self.training_config = training_config or {}
        
        # Validate configuration
        self._validate_config()
        
        # Create MONAI UNet with configuration parameters
        self.unet = UNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.features,
            strides=[2] * (len(config.features) - 1),  # Standard downsampling
            num_res_units=2,  # Standard residual units per block
            act=config.activation,
            norm="BATCH",  # Batch normalization for stable training
            dropout=0.0,  # No dropout by default
        )
        
        logger.info(f"UNet2D initialized with config: {config}")
        logger.info(
            f"Architecture: {config.spatial_dims}D, "
            f"channels: {config.in_channels}→{config.out_channels}, "
            f"features: {config.features}, activation: {config.activation}"
        )
        
    def _validate_config(self) -> None:
        """Validate model configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.spatial_dims != 2:
            raise ValueError(
                f"UNet2D only supports 2D processing, got spatial_dims={self.config.spatial_dims}"
            )
            
        if self.config.in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {self.config.in_channels}"
            )
            
        if self.config.out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {self.config.out_channels}"
            )
            
        if len(self.config.features) < 2:
            raise ValueError(
                f"features must have at least 2 levels, got {len(self.config.features)}"
            )
            
        if len(self.config.img_size) != 2:
            raise ValueError(
                f"img_size must be 2D [H, W], got {self.config.img_size}"
            )
            
        logger.debug("Model configuration validation passed")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet.
        
        Args:
            x: Input tensor of shape [N, C, H, W]
            
        Returns:
            Output segmentation tensor of shape [N, C, H, W]
            
        Raises:
            ValueError: If input tensor shape is invalid
        """
        # Validate input tensor
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor [N, C, H, W], got {x.dim()}D tensor"
            )
            
        batch_size, channels, height, width = x.shape
        
        if channels != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {channels}"
            )
            
        # Validate input dimensions match configuration
        expected_h, expected_w = self.config.img_size
        if height != expected_h or width != expected_w:
            logger.warning(
                f"Input size ({height}, {width}) differs from configured size "
                f"({expected_h}, {expected_w}). Model may not perform optimally."
            )
        
        # Forward pass through MONAI UNet
        output = self.unet(x)
        
        # Validate output shape
        self._validate_output_shape(output, x.shape)
        
        return output
    
    def _validate_output_shape(self, output: torch.Tensor, input_shape: torch.Size) -> None:
        """Validate that output tensor has correct shape.
        
        Args:
            output: Model output tensor
            input_shape: Original input tensor shape
            
        Raises:
            RuntimeError: If output shape is unexpected
        """
        expected_shape = (
            input_shape[0],  # Same batch size
            self.config.out_channels,  # Configured output channels
            input_shape[2],  # Same height
            input_shape[3],  # Same width
        )
        
        if output.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected output shape {output.shape}, expected {expected_shape}"
            )
            
        logger.debug(f"Output shape validation passed: {output.shape}")

    def configure_optimizers(self) -> Any:
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer or dictionary containing optimizer and scheduler configuration
        """
        from torch.optim import Adam, AdamW, SGD
        from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
        
        # Get training configuration from training_config or use defaults
        optimizer_name = self.training_config.get('optimizer', 'AdamW')
        learning_rate = self.training_config.get('learning_rate', 1e-4)
        scheduler_name = self.training_config.get('scheduler', None)
        scheduler_kwargs = self.training_config.get('scheduler_kwargs', {})
        
        # Initialize optimizer
        if optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-2)
        elif optimizer_name.lower() == "sgd":
            optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Available: 'Adam', 'AdamW', 'SGD'")
        
        logger.info(f"Configured {optimizer_name} optimizer with lr={learning_rate}")
        
        # Return optimizer only if no scheduler is specified
        if not scheduler_name:
            return optimizer
        
        # Configure scheduler
        scheduler_name_lower = scheduler_name.lower()
        if scheduler_name_lower == "steplr":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_kwargs.get("step_size", 30),
                gamma=scheduler_kwargs.get("gamma", 0.1),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif scheduler_name_lower == "cosineannealinglr":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_kwargs.get("T_max", 100),
                eta_min=scheduler_kwargs.get("eta_min", 1e-6),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        elif scheduler_name_lower == "reducelronplateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_kwargs.get("mode", "min"),
                factor=scheduler_kwargs.get("factor", 0.5),
                patience=scheduler_kwargs.get("patience", 10),
                min_lr=scheduler_kwargs.get("min_lr", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_kwargs.get("monitor", "val_loss"),
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        else:
            raise ValueError(f"Unknown scheduler '{scheduler_name}'. Available: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step compatible with volume-based data pipeline.
        
        Args:
            batch: Training batch with keys 'volume' and 'segmentation'
            batch_idx: Batch index
            
        Returns:
            Training loss tensor
        """
        from niftilearn.training.losses import get_loss_function
        from niftilearn.training.metrics import SegmentationMetrics
        
        # Initialize loss function if not already done
        if not hasattr(self, 'criterion'):
            loss_name = self.training_config.get('loss_function', 'dicece')
            loss_kwargs = self.training_config.get('loss_kwargs', {})
            self.criterion = get_loss_function(loss_name, **loss_kwargs)
            
        # Initialize training metrics if not already done
        if not hasattr(self, 'train_metrics'):
            self.train_metrics = SegmentationMetrics(include_background=True, reduction="mean")
        
        # Extract data from batch (using correct keys from volume_batch_collate)
        images = batch["volume"]        # [N, C, H, W] where N = num_slices
        targets = batch["segmentation"] # [N, C, H, W]
        
        # Forward pass
        predictions = self.forward(images)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Update training metrics (convert predictions to probabilities for metrics)
        with torch.no_grad():
            pred_probs = torch.sigmoid(predictions) if predictions.min() < 0 else predictions
            self.train_metrics.update(pred_probs, targets)
        
        # Log training loss (Lightning 2.x pattern)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step compatible with volume-based data pipeline.
        
        Args:
            batch: Validation batch with keys 'volume' and 'segmentation'
            batch_idx: Batch index
            
        Returns:
            Validation loss tensor
        """
        from niftilearn.training.losses import get_loss_function
        from niftilearn.training.metrics import SegmentationMetrics
        
        # Initialize loss function if not already done
        if not hasattr(self, 'criterion'):
            loss_name = self.training_config.get('loss_function', 'dicece')
            loss_kwargs = self.training_config.get('loss_kwargs', {})
            self.criterion = get_loss_function(loss_name, **loss_kwargs)
            
        # Initialize validation metrics if not already done
        if not hasattr(self, 'val_metrics'):
            self.val_metrics = SegmentationMetrics(include_background=True, reduction="mean")
        
        # Extract data from batch (using correct keys from volume_batch_collate)
        images = batch["volume"]        # [N, C, H, W]
        targets = batch["segmentation"] # [N, C, H, W]
        
        # Forward pass
        predictions = self.forward(images)
        
        # Calculate loss
        loss = self.criterion(predictions, targets)
        
        # Update validation metrics (convert predictions to probabilities for metrics)
        pred_probs = torch.sigmoid(predictions) if predictions.min() < 0 else predictions
        self.val_metrics.update(pred_probs, targets)
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch (Lightning 2.x)."""
        if hasattr(self, 'train_metrics'):
            # Compute and log training metrics
            train_metrics = self.train_metrics.compute()
            
            for metric_name, metric_value in train_metrics.items():
                # Handle both scalar and tensor metric values
                if metric_value.numel() == 1:
                    metric_value = metric_value.item()
                else:
                    # For multi-class metrics, log mean
                    metric_value = metric_value.mean().item()
                
                self.log(f"train_{metric_name}", metric_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            # Reset metrics for next epoch
            self.train_metrics.reset()
        
        # Log learning rate if available
        if hasattr(self, 'trainer') and self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("learning_rate", current_lr, on_epoch=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch (Lightning 2.x)."""
        if hasattr(self, 'val_metrics'):
            # Compute and log validation metrics
            val_metrics = self.val_metrics.compute()
            
            for metric_name, metric_value in val_metrics.items():
                # Handle both scalar and tensor metric values
                if metric_value.numel() == 1:
                    metric_value = metric_value.item()
                else:
                    # For multi-class metrics, log mean
                    metric_value = metric_value.mean().item()
                
                self.log(f"val_{metric_name}", metric_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            # Reset metrics for next epoch
            self.val_metrics.reset()

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step for inference.
        
        Args:
            batch: Batch containing 'volume' tensor
            batch_idx: Batch index
            
        Returns:
            Model predictions as probabilities
        """
        images = batch["volume"]
        predictions = self.forward(images)
        
        # Convert to probabilities
        probabilities = torch.sigmoid(predictions) if predictions.min() < 0 else predictions
        
        return probabilities

    def get_model_info(self) -> dict[str, Any]:
        """Get model architecture information.
        
        Returns:
            Dictionary containing model configuration and basic info
        """
        return {
            "model_type": "UNet2D",
            "spatial_dims": self.config.spatial_dims,
            "in_channels": self.config.in_channels,
            "out_channels": self.config.out_channels,
            "features": self.config.features,
            "activation": self.config.activation,
            "img_size": self.config.img_size,
            "slice_axis": self.config.slice_axis,
        }