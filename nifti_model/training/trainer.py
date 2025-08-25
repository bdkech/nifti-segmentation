"""Training module using PyTorch Lightning."""

from pathlib import Path
from typing import Optional

import lightning
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger

from ..data.loaders import NiftiDataModule
from ..models.unetr import UNet2DModel
from ..utils.config import Config
from .logging import setup_wandb


class TrainingManager:
    """Manager class for training 2D UNet models with Lightning.
    
    Handles:
    - Training setup and execution for slice-based training
    - Callback configuration
    - Logging setup
    - Checkpoint management
    - Multi-GPU training coordination
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize training manager.
        
        Args:
            config: Complete training configuration
        """
        self.config = config
        self.trainer: Optional[lightning.Trainer] = None
        self.model: Optional[UNet2DModel] = None
        self.data_module: Optional[NiftiDataModule] = None
        
        # Ensure output directories exist
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training manager initialized with config: {config}")
    
    def setup_model(self) -> UNet2DModel:
        """Initialize and return the 2D UNet model.
        
        Returns:
            Configured 2D UNet model
        """
        self.model = UNet2DModel(
            model_config=self.config.model,
            training_config=self.config.training,
        )
        
        logger.info("2D UNet model initialized successfully")
        logger.debug(self.model.get_model_summary())
        
        return self.model
    
    def setup_data(self) -> NiftiDataModule:
        """Initialize and return the data module.
        
        Returns:
            Configured data module
        """
        self.data_module = NiftiDataModule(config=self.config.data)
        
        logger.info("Data module initialized successfully")
        return self.data_module
    
    def setup_callbacks(self) -> list[lightning.Callback]:
        """Create and configure training callbacks.
        
        Returns:
            List of configured callbacks
        """
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename="unet2d-{epoch:02d}-{val/dice:.4f}",
            monitor=self.config.training.val_metric,
            mode="max",  # Maximize validation dice
            save_top_k=self.config.training.save_top_k,
            save_last=True,
            every_n_epochs=self.config.training.checkpoint_every_n_epochs,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.training.patience:
            early_stopping = EarlyStopping(
                monitor=self.config.training.val_metric,
                mode="max",
                patience=self.config.training.patience,
                min_delta=self.config.training.min_delta,
                verbose=True,
            )
            callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(
            logging_interval="epoch",
            log_momentum=True,
        )
        callbacks.append(lr_monitor)
        
        # Rich progress bar
        progress_bar = RichProgressBar(
            leave=True,
        )
        callbacks.append(progress_bar)
        
        logger.info(f"Configured {len(callbacks)} training callbacks")
        return callbacks
    
    def setup_logger(self) -> Optional[WandbLogger]:
        """Setup Weights & Biases logger if enabled.
        
        Returns:
            WandbLogger instance if enabled, None otherwise
        """
        if not self.config.wandb.enabled:
            logger.info("W&B logging disabled")
            return None
        
        # Setup W&B integration
        setup_wandb(self.config.wandb)
        
        # Create logger
        wandb_logger = WandbLogger(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
            notes=self.config.wandb.notes,
            log_model=self.config.wandb.log_model,
            save_dir=str(self.config.log_dir),
        )
        
        logger.info(f"W&B logger initialized for project: {self.config.wandb.project}")
        return wandb_logger
    
    def setup_trainer(
        self,
        callbacks: list[lightning.Callback],
        wandb_logger: Optional[WandbLogger] = None,
    ) -> lightning.Trainer:
        """Configure and create Lightning trainer.
        
        Args:
            callbacks: List of training callbacks
            wandb_logger: Optional W&B logger
            
        Returns:
            Configured Lightning trainer
        """
        # Configure strategy for multi-GPU training
        strategy = "auto"
        if self.config.devices and self.config.devices > 1:
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        
        # Create trainer
        self.trainer = lightning.Trainer(
            # Training configuration
            max_epochs=self.config.training.epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            strategy=strategy,
            precision=self.config.precision,
            
            # Validation configuration
            check_val_every_n_epoch=self.config.training.val_interval,
            
            # Logging and callbacks
            logger=wandb_logger,
            callbacks=callbacks,
            log_every_n_steps=50,
            
            # Performance optimizations
            gradient_clip_val=self.config.training.grad_clip_val,
            accumulate_grad_batches=1,
            
            # Reproducibility
            deterministic=False,  # Set to True for full reproducibility
            
            # Other settings
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=True,
        )
        
        logger.info("Lightning trainer configured successfully")
        return self.trainer
    
    def train(
        self,
        resume_from_checkpoint: Optional[Path] = None,
    ) -> None:
        """Execute training process.
        
        Args:
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """
        logger.info("Starting training process...")
        
        # Setup components
        model = self.setup_model()
        data_module = self.setup_data()
        callbacks = self.setup_callbacks()
        wandb_logger = self.setup_logger()
        trainer = self.setup_trainer(callbacks, wandb_logger)
        
        # Log configuration to W&B
        if wandb_logger:
            wandb_logger.experiment.config.update(self.config.dict())
        
        # Start training
        try:
            trainer.fit(
                model=model,
                datamodule=data_module,
                ckpt_path=str(resume_from_checkpoint) if resume_from_checkpoint else None,
            )
            
            logger.info("Training completed successfully!")
            
            # Log best metrics
            if hasattr(trainer, "callback_metrics"):
                best_dice = trainer.callback_metrics.get("val/dice", 0.0)
                logger.info(f"Best validation Dice: {best_dice:.4f}")
        
        except Exception as e:
            logger.exception("Training failed")
            raise e
        
        finally:
            # Cleanup
            if wandb_logger:
                wandb_logger.experiment.finish()
    
    def validate(
        self,
        checkpoint_path: Path,
        data_module: Optional[NiftiDataModule] = None,
    ) -> dict[str, float]:
        """Run validation on a trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            data_module: Optional data module (will create if None)
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Running validation with checkpoint: {checkpoint_path}")
        
        # Load model from checkpoint
        model = UNet2DModel.load_from_checkpoint(
            str(checkpoint_path),
            model_config=self.config.model,
            training_config=self.config.training,
        )
        
        # Setup data if not provided
        if data_module is None:
            data_module = self.setup_data()
        
        # Create trainer for validation
        trainer = lightning.Trainer(
            accelerator=self.config.accelerator,
            devices=1,  # Use single device for validation
            logger=False,
            enable_progress_bar=True,
        )
        
        # Run validation
        results = trainer.validate(model=model, datamodule=data_module)
        
        logger.info("Validation completed")
        return results[0] if results else {}
    
    def test(
        self,
        checkpoint_path: Path,
        test_data_dir: Path,
    ) -> dict[str, float]:
        """Run testing on a trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_data_dir: Path to test data directory
            
        Returns:
            Dictionary of test metrics
        """
        logger.info(f"Running testing with checkpoint: {checkpoint_path}")
        
        # Load model from checkpoint
        model = UNet2DModel.load_from_checkpoint(
            str(checkpoint_path),
            model_config=self.config.model,
            training_config=self.config.training,
        )
        
        # Create test data module
        test_config = self.config.data.copy()
        test_config.data_dir = test_data_dir
        test_data_module = NiftiDataModule(config=test_config)
        
        # Create trainer for testing
        trainer = lightning.Trainer(
            accelerator=self.config.accelerator,
            devices=1,
            logger=False,
            enable_progress_bar=True,
        )
        
        # Run testing
        results = trainer.test(model=model, datamodule=test_data_module)
        
        logger.info("Testing completed")
        return results[0] if results else {}


def create_trainer_from_config(config: Config) -> TrainingManager:
    """Create training manager from configuration.
    
    Args:
        config: Complete training configuration
        
    Returns:
        Initialized training manager
    """
    return TrainingManager(config)