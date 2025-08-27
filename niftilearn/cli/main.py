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

    # TODO: Implement training logic
    logger.warning("Training implementation not yet available")


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

    # TODO: Implement prediction logic
    logger.warning("Prediction implementation not yet available")


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
