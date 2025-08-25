"""Command line interface for NIFTI model segmentation pipeline."""

from pathlib import Path
from typing import Optional

import click
from loguru import logger

from ..training.trainer import TrainingManager
from ..utils.config import Config, DataConfig, ModelConfig, TrainingConfig, WandbConfig
from ..utils.logging import setup_logging


def build_config_from_args(
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Config:
    """Build configuration object from CLI arguments.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save outputs
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        Complete configuration object with defaults
    """
    # Create data configuration
    data_config = DataConfig(
        data_dir=data_dir,
        annotation_dir=data_dir,  # Assume annotations in same directory for now
        train_split=0.8,
        val_split=0.15,
        test_split=0.05,
        volume_pattern="*.nii.gz",
        annotation_pattern="*_seg.nii.gz",
        slice_axis=2,  # Sagittal slices by default
        batch_size=batch_size,
        img_size=[224, 224],
    )
    
    # Create model configuration  
    model_config = ModelConfig(
        img_size=[224, 224],
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32),
        activation="PRELU",
        norm_name="INSTANCE",
        dropout_rate=0.0,
        spatial_dims=2,
        slice_axis=2,
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=1e-5,
        optimizer="AdamW",
        scheduler=None,
        loss_function="DiceCELoss",
        val_interval=1,
        val_metric="val/dice",
        save_top_k=3,
        checkpoint_every_n_epochs=10,
    )
    
    # Create W&B configuration (disabled by default for CLI)
    wandb_config = WandbConfig(
        enabled=False,  # Keep simple for CLI usage
        project="nifti-segmentation-cli",
        tags=["cli", "unet2d"],
    )
    
    # Create complete configuration
    config = Config(
        data=data_config,
        model=model_config,
        training=training_config,
        wandb=wandb_config,
        output_dir=output_dir,
        accelerator="auto",
        devices=None,  # Auto-detect
        num_workers=4,
        precision="32",
    )
    
    # Set up derived paths
    config.checkpoint_dir = output_dir / "checkpoints"
    config.log_dir = output_dir / "logs"
    
    return config


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", "-d", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(
    ctx: click.Context,
    config: Optional[Path],
    verbose: bool,
    debug: bool,
) -> None:
    """NIFTI volume segmentation pipeline using UNetR architecture.
    
    This tool processes medical NIFTI volumes for segmentation tasks by:
    - Splitting 3D volumes into 2D slices
    - Training UNetR models on slice-based data
    - Reconstructing full volume predictions
    
    Commands can be chained together like Docker commands.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return
    
    # Setup logging based on verbosity
    setup_logging(verbose=verbose, debug=debug)
    
    # Load configuration
    if config:
        # TODO: Load config from file
        pass
    else:
        # TODO: Use default config
        pass


@main.command()
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing NIFTI volumes and annotations",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save preprocessed data",
)
@click.option(
    "--train-split",
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    help="Fraction of data for training (default: 0.8)",
)
def preprocess(
    data_dir: Path,
    output_dir: Path,
    train_split: float,
) -> None:
    """Preprocess NIFTI volumes and create training datasets.
    
    This command:
    - Loads NIFTI volumes and corresponding annotations
    - Applies preprocessing transforms (normalization, etc.)
    - Splits data into training/validation sets
    - Saves preprocessed slices for training
    """
    # TODO: Implement preprocessing pipeline
    click.echo(f"Preprocessing data from {data_dir} to {output_dir}")
    click.echo(f"Train split: {train_split}")


@main.command()
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing preprocessed training data",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save model checkpoints",
)
@click.option(
    "--epochs", "-e", type=int, default=100, help="Number of training epochs"
)
@click.option(
    "--batch-size", "-b", type=int, default=8, help="Batch size (slices per batch from same volume)"
)
@click.option(
    "--lr", type=float, default=1e-4, help="Learning rate"
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    help="Path to checkpoint to resume training from",
)
def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    resume: Optional[Path],
) -> None:
    """Train UNetR model on preprocessed NIFTI data.
    
    This command:
    - Loads preprocessed slice data
    - Initializes UNetR architecture
    - Trains model with Lightning framework
    - Logs progress to Weights & Biases
    """
    logger.info(f"Starting training with data from {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
    
    try:
        # Validate input directory
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        log_dir = output_dir / "logs"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directories: {output_dir}")
        
        # Build configuration from CLI arguments
        config = build_config_from_args(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        
        logger.info("Configuration built successfully")
        
        # Initialize training manager
        trainer_manager = TrainingManager(config)
        logger.info("Training manager initialized")
        
        # Start training
        trainer_manager.train(resume_from_checkpoint=resume)
        
        logger.info("Training completed successfully!")
        click.echo("✅ Training completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.ClickException(str(e))
        
    except Exception as e:
        logger.exception("Training failed with unexpected error")
        click.echo(f"❌ Training failed: {e}", err=True)
        raise click.ClickException(f"Training failed: {e}")


@main.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model checkpoint",
)
@click.option(
    "--input-volume",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input NIFTI volume for prediction",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save prediction volume",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=16,
    help="Batch size for inference",
)
def predict(
    model_path: Path,
    input_volume: Path,
    output_path: Path,
    batch_size: int,
) -> None:
    """Generate segmentation predictions for a NIFTI volume.
    
    This command:
    - Loads trained model checkpoint
    - Processes input volume slice by slice
    - Reconstructs full 3D segmentation volume
    - Saves result as NIFTI file
    """
    # TODO: Implement inference pipeline
    click.echo(f"Generating predictions with model: {model_path}")
    click.echo(f"Input volume: {input_volume}")
    click.echo(f"Output path: {output_path}")
    click.echo(f"Batch size: {batch_size}")


@main.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model checkpoint",
)
@click.option(
    "--test-data",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing test volumes and annotations",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save evaluation results",
)
def evaluate(
    model_path: Path,
    test_data: Path,
    output_dir: Optional[Path],
) -> None:
    """Evaluate model performance on test dataset.
    
    This command:
    - Loads trained model and test data
    - Generates predictions for test volumes
    - Computes segmentation metrics (Dice, IoU, etc.)
    - Optionally saves detailed results
    """
    # TODO: Implement evaluation pipeline
    click.echo(f"Evaluating model: {model_path}")
    click.echo(f"Test data: {test_data}")
    if output_dir:
        click.echo(f"Results will be saved to: {output_dir}")


if __name__ == "__main__":
    main()