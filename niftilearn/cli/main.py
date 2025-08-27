"""Command-line interface for NiftiLearn."""

from pathlib import Path
from typing import Optional

import click
from loguru import logger

from niftilearn.core.logging import setup_logging


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration YAML file",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    help="Path to log file (optional)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress all output except errors",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Enable debug output",
)
@click.pass_context
def main(
    ctx: click.Context,
    config: Optional[Path],
    log_file: Optional[Path],
    quiet: bool,
    verbose: bool,
    debug: bool,
) -> None:
    """NiftiLearn: NIFTI volume segmentation using slice-based UNet architecture."""
    # Determine log level from flags
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    elif quiet:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    # Setup logging
    setup_logging(level=log_level, log_file=log_file)

    # Store config and log settings in context
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["log_level"] = log_level

    logger.info(f"NiftiLearn CLI started with log level: {log_level}")


@main.command()
@click.option(
    "--epochs",
    "-e",
    type=int,
    help="Number of training epochs (overrides config)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    help="Training batch size (overrides config)",
)
@click.option(
    "--learning-rate",
    "-lr",
    type=float,
    help="Learning rate (overrides config)",
)
@click.pass_context
def train(
    ctx: click.Context,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
) -> None:
    """Train a segmentation model on NIFTI volumes."""
    config_path = ctx.obj.get("config_path")

    if not config_path:
        raise click.ClickException(
            "Configuration file required for training. Use --config option."
        )

    logger.info(f"Starting training with config: {config_path}")

    # CLI overrides
    overrides = {}
    if epochs is not None:
        overrides["epochs"] = epochs
        logger.info(f"Overriding epochs: {epochs}")
    if batch_size is not None:
        overrides["batch_size"] = batch_size
        logger.info(f"Overriding batch size: {batch_size}")
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
        logger.info(f"Overriding learning rate: {learning_rate}")

    # Load configuration
    from niftilearn.config.loader import load_config
    from niftilearn.data.datamodule import NiftiDataModule
    from niftilearn.models.unet import UNet2D
    import pytorch_lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
    
    try:
        config = load_config(config_path)
        
        # Apply CLI overrides to training config
        if epochs is not None:
            config.training.epochs = epochs
        if batch_size is not None:
            config.training.batch_size = batch_size
            config.data.batch_size = batch_size  # Keep data and training in sync
        if learning_rate is not None:
            config.training.learning_rate = learning_rate
            
        logger.info(f"Training configuration loaded: {config.training}")
        
        # Create model and data module
        training_config_dict = {
            'optimizer': config.training.optimizer,
            'learning_rate': config.training.learning_rate,
            'loss_function': config.training.loss_function,
            'loss_kwargs': config.training.loss_kwargs,
            'scheduler': config.training.scheduler,
            'scheduler_kwargs': config.training.scheduler_kwargs,
        }
        model = UNet2D(config.model, training_config_dict)
        datamodule = NiftiDataModule(config.data)
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.output_dir / "checkpoints",
            filename="nifti-unet-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if config.training.patience > 0:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.training.patience,
                min_delta=config.training.min_delta,
                verbose=True,
            )
            callbacks.append(early_stopping)
        
        # Progress bar
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            accelerator=config.compute.accelerator,
            devices=config.compute.devices,
            precision=config.compute.precision,
            callbacks=callbacks,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=50,
        )
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {config.training.epochs} epochs")
        logger.info(f"Model: {model.get_model_info()}")
        
        # Start training
        trainer.fit(model, datamodule=datamodule)
        
        # Log results
        best_checkpoint = checkpoint_callback.best_model_path
        best_score = checkpoint_callback.best_model_score
        
        logger.info("Training completed successfully!")
        logger.info(f"Best checkpoint: {best_checkpoint}")
        logger.info(f"Best validation loss: {best_score}")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise click.ClickException(f"Training failed: {e}")


@main.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model checkpoint",
)
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input NIFTI volume",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to output segmentation volume",
)
@click.pass_context
def predict(
    ctx: click.Context,
    model: Path,
    input: Path,
    output: Path,
) -> None:
    """Generate predictions on a NIFTI volume using a trained model."""
    config_path = ctx.obj.get("config_path")

    logger.info(f"Starting prediction with model: {model}")
    logger.info(f"Input volume: {input}")
    logger.info(f"Output path: {output}")

    if config_path:
        logger.info(f"Using config: {config_path}")

    # Load model from checkpoint
    from niftilearn.models.unet import UNet2D
    from niftilearn.data.transforms import VolumeSliceExtractor
    from niftilearn.utils.reconstruction import reconstruct_volume_from_predictions
    import torch
    import nibabel as nib
    import numpy as np
    
    try:
        # Load trained model
        logger.info(f"Loading model from checkpoint: {model}")
        trained_model = UNet2D.load_from_checkpoint(str(model))
        trained_model.eval()
        
        # Load configuration if available
        slice_axis = 2  # Default
        img_size = [224, 224]  # Default
        
        if config_path:
            from niftilearn.config.loader import load_config
            config = load_config(config_path)
            slice_axis = config.model.slice_axis
            img_size = config.model.img_size
            logger.info(f"Using config slice_axis={slice_axis}, img_size={img_size}")
        
        # Load input volume
        logger.info(f"Loading input volume: {input}")
        volume_img = nib.load(str(input))
        volume_data = volume_img.get_fdata()
        
        logger.info(f"Input volume shape: {volume_data.shape}")
        
        # Extract slices for processing
        slice_extractor = VolumeSliceExtractor(
            slice_axis=slice_axis,
            target_size=img_size,
            target_spacing=[1.0, 1.0, 1.0],  # Default spacing
        )
        
        # Process volume to extract slices
        processed_slices = slice_extractor.extract_slices(volume_data)
        
        # Convert to tensor and add batch dimension
        slice_tensor = torch.from_numpy(processed_slices).float()
        if slice_tensor.dim() == 3:
            slice_tensor = slice_tensor.unsqueeze(1)  # Add channel dimension [N, C, H, W]
        
        logger.info(f"Processing {slice_tensor.shape[0]} slices")
        
        # Run inference
        predictions_list = []
        with torch.no_grad():
            # Process in batches to handle memory efficiently
            batch_size = 8
            for i in range(0, slice_tensor.shape[0], batch_size):
                batch = slice_tensor[i:i + batch_size]
                batch_predictions = trained_model(batch)
                
                # Convert to probabilities
                batch_probs = torch.sigmoid(batch_predictions)
                predictions_list.append(batch_probs)
        
        # Concatenate all predictions
        all_predictions = torch.cat(predictions_list, dim=0)
        
        # Convert probabilities to binary masks (threshold at 0.5)
        binary_predictions = (all_predictions > 0.5).float()
        
        logger.info(f"Generated predictions with shape: {binary_predictions.shape}")
        
        # Create slice indices (assuming all slices in order)
        slice_indices = list(range(binary_predictions.shape[0]))
        
        # Reconstruct 3D volume
        logger.info("Reconstructing 3D volume from slice predictions")
        output_volume_path = reconstruct_volume_from_predictions(
            predictions=binary_predictions,
            slice_indices=slice_indices,
            reference_volume=input,
            output_path=output,
            slice_axis=slice_axis,
        )
        
        logger.info(f"Prediction completed successfully!")
        logger.info(f"Output segmentation saved: {output_volume_path}")
        
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise click.ClickException(f"Prediction failed: {e}")


@main.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model checkpoint",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    help="Path to validation data directory",
)
@click.pass_context
def validate(
    ctx: click.Context,
    model: Path,
    data_dir: Optional[Path],
) -> None:
    """Validate a trained model on test data."""
    config_path = ctx.obj.get("config_path")

    if not config_path and not data_dir:
        raise click.ClickException(
            "Either configuration file (--config) or data directory (--data-dir) required."
        )

    logger.info(f"Starting validation with model: {model}")

    if config_path:
        logger.info(f"Using config: {config_path}")
    if data_dir:
        logger.info(f"Using data directory: {data_dir}")

    # TODO: Implement validation logic
    logger.warning("Validation implementation not yet available")


if __name__ == "__main__":
    main()
