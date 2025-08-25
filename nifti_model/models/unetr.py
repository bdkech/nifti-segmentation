"""2D UNet model wrapper using MONAI architecture for slice-based training."""

from typing import Any, Optional

import lightning
import torch
import torch.nn as nn
from loguru import logger
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import BasicUNet
from monai.transforms import Activations, AsDiscrete, Compose
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..utils.config import ModelConfig, TrainingConfig


class UNet2DModel(lightning.LightningModule):
    """Lightning module wrapping MONAI BasicUNet for 2D slice segmentation.
    
    This module handles:
    - 2D UNet initialization with configurable parameters
    - Training and validation steps with slice-based metrics
    - Optimizer and scheduler configuration
    - Inference on individual slices
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        """Initialize 2D UNet model.
        
        Args:
            model_config: Model architecture configuration
            training_config: Training parameters configuration
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.training_config = training_config
        
        # Initialize 2D UNet model
        self.model = BasicUNet(
            spatial_dims=2,  # Force 2D for slice processing
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            features=model_config.features,
            act=model_config.activation,
            norm=model_config.norm_name,
            bias=True,
            dropout=model_config.dropout_rate,
        )
        
        # Initialize loss function
        self.loss_function = self._create_loss_function()
        
        # Initialize metrics
        self.train_dice = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )
        self.val_dice = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )
        
        # Optional additional metrics
        self.val_hd = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )
        
        # Post-processing transforms
        self.post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ])
        self.post_label = Compose([
            AsDiscrete(to_onehot=model_config.out_channels),
        ])
        
        # Store best validation metric
        self.best_val_dice = 0.0
        
        logger.info(f"Initialized 2D UNet model with {self.count_parameters()} parameters")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration.
        
        Returns:
            Configured loss function
        """
        loss_name = self.training_config.loss_function
        loss_kwargs = self.training_config.loss_kwargs or {}
        
        if loss_name == "DiceCELoss":
            return DiceCELoss(
                include_background=False,
                to_onehot_y=True,
                sigmoid=True,
                **loss_kwargs,
            )
        elif loss_name == "DiceLoss":
            return DiceLoss(
                include_background=False,
                to_onehot_y=True,
                sigmoid=True,
                **loss_kwargs,
            )
        elif loss_name == "FocalLoss":
            return FocalLoss(**loss_kwargs)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W) for 2D slices
            
        Returns:
            Model predictions of shape (B, C, H, W)
        """
        return self.model(x)
    
    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch containing 'image' and 'label' tensors
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        images = batch["image"]
        labels = batch["label"]
        
        # Forward pass
        outputs = self.forward(images)
        
        # Compute loss
        loss = self.loss_function(outputs, labels)
        
        # Compute metrics
        with torch.no_grad():
            outputs_post = self.post_pred(outputs)
            labels_post = self.post_label(labels)
            self.train_dice(outputs_post, labels_post)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/dice",
            self.train_dice.aggregate().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return loss
    
    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Validation step for 2D slices.
        
        Args:
            batch: Batch containing 'image' and 'label' tensors (2D slices)
            batch_idx: Batch index
            
        Returns:
            Dictionary containing loss and predictions
        """
        images = batch["image"]
        labels = batch["label"]
        
        # Forward pass (no sliding window needed for 2D slices)
        outputs = self.forward(images)
        
        # Compute loss
        loss = self.loss_function(outputs, labels)
        
        # Post-process for metrics
        outputs_post = self.post_pred(outputs)
        labels_post = self.post_label(labels)
        
        # Update metrics
        self.val_dice(outputs_post, labels_post)
        self.val_hd(outputs_post, labels_post)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            "val_loss": loss,
            "val_preds": outputs_post,
            "val_labels": labels_post,
        }
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch to aggregate metrics."""
        # Aggregate dice metric
        mean_dice = self.val_dice.aggregate().item()
        self.log("val/dice", mean_dice, prog_bar=True)
        
        # Aggregate Hausdorff distance
        try:
            mean_hd = self.val_hd.aggregate().item()
            self.log("val/hausdorff", mean_hd)
        except Exception as e:
            logger.warning(f"Could not compute Hausdorff distance: {e}")
        
        # Update best validation score
        if mean_dice > self.best_val_dice:
            self.best_val_dice = mean_dice
            self.log("val/best_dice", self.best_val_dice)
        
        # Reset metrics for next epoch
        self.val_dice.reset()
        self.val_hd.reset()
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Reset training metrics
        self.train_dice.reset()
    
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and optionally scheduler configuration
        """
        # Create optimizer
        optimizer = self._create_optimizer()
        
        config = {"optimizer": optimizer}
        
        # Add scheduler if specified
        if self.training_config.scheduler:
            scheduler = self._create_scheduler(optimizer)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": self.training_config.val_metric,
                "interval": "epoch",
                "frequency": 1,
            }
        
        return config
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration.
        
        Returns:
            Configured optimizer
        """
        optimizer_name = self.training_config.optimizer
        
        if optimizer_name == "AdamW":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        elif optimizer_name == "Adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        elif optimizer_name == "SGD":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        """Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to attach scheduler to
            
        Returns:
            Configured scheduler or None
        """
        scheduler_name = self.training_config.scheduler
        
        if scheduler_name == "CosineAnnealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.epochs,
                eta_min=1e-6,
            )
        elif scheduler_name == "StepLR":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif scheduler_name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",  # Maximize dice score
                factor=0.5,
                patience=10,
                verbose=True,
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}")
            return None
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary.
        
        Returns:
            String containing model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
2D UNet Model Summary:
---------------------
Input Size: {self.model_config.img_size}
Input Channels: {self.model_config.in_channels}
Output Channels: {self.model_config.out_channels}
Features: {self.model_config.features}
Activation: {self.model_config.activation}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}
"""
        return summary.strip()
    
    def predict_slice(
        self,
        slice_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Predict segmentation for a 2D slice.
        
        Args:
            slice_tensor: Input slice tensor (1, C, H, W)
            
        Returns:
            Predicted segmentation slice
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(slice_tensor)
            
            # Apply post-processing
            predictions = self.post_pred(predictions)
        
        return predictions
    
    def predict_volume_from_slices(
        self,
        slices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict segmentation for a full volume by processing individual slices.
        
        Args:
            slices: List of 2D slice tensors (each shaped as (1, C, H, W))
            
        Returns:
            Predicted segmentation volume reconstructed from slices
        """
        self.eval()
        predicted_slices = []
        
        with torch.no_grad():
            for slice_tensor in slices:
                pred_slice = self.predict_slice(slice_tensor)
                predicted_slices.append(pred_slice)
        
        # Stack slices to reconstruct volume (B, C, H, W, D)
        volume_prediction = torch.stack(predicted_slices, dim=-1)
        
        return volume_prediction