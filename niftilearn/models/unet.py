"""UNet2D model implementation using MONAI for medical image segmentation."""

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from monai.networks.nets import UNet

from niftilearn.config.models import ModelConfig


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

    def __init__(self, config: ModelConfig) -> None:
        """Initialize UNet2D model.
        
        Args:
            config: Model configuration containing architecture parameters
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        super().__init__()
        
        self.config = config
        
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
        """Configure optimizer for training.
        
        This method will be implemented in the training module (Task 10).
        For now, it raises NotImplementedError to indicate the model
        is not yet ready for training.
        
        Returns:
            NotImplementedError: Training not yet implemented
        """
        raise NotImplementedError(
            "Optimizer configuration will be implemented in the training module (Task 10)"
        )

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """Training step implementation.
        
        This method will be implemented in the training module (Task 10).
        
        Args:
            batch: Training batch data
            batch_idx: Batch index
            
        Returns:
            NotImplementedError: Training not yet implemented
        """
        raise NotImplementedError(
            "Training step will be implemented in the training module (Task 10)"
        )

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """Validation step implementation.
        
        This method will be implemented in the training module (Task 10).
        
        Args:
            batch: Validation batch data
            batch_idx: Batch index
            
        Returns:
            NotImplementedError: Training not yet implemented
        """
        raise NotImplementedError(
            "Validation step will be implemented in the training module (Task 10)"
        )

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