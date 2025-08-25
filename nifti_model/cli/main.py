"""Command line interface for NIFTI model segmentation pipeline."""

from pathlib import Path
from typing import Optional

import click

from ..utils.logging import setup_logging


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
    "--batch-size", "-b", type=int, default=1, help="Batch size (volumes per batch)"
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
    # TODO: Implement training loop
    click.echo(f"Training model with data from {data_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    if resume:
        click.echo(f"Resuming from checkpoint: {resume}")


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