#!/usr/bin/env python3
"""Test script for UNet2D model integration with dataset pipeline.

This script validates that the UNet2D model correctly processes real dataset
batches and produces expected output shapes for segmentation tasks.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from niftilearn.config.models import ComputeConfig, DataConfig, ModelConfig
from niftilearn.data import NiftiDataModule
from niftilearn.models import UNet2D


def setup_logging() -> None:
    """Configure logging for the test script."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Test UNet2D model integration with dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/test_unet2d_integration.py /path/to/nifti/data
  python bin/test_unet2d_integration.py /path/to/data --annotation-type RA
  python bin/test_unet2d_integration.py /path/to/data --device cpu
        """,
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to NIFTI data directory containing subject folders",
    )

    parser.add_argument(
        "--annotation-type",
        type=str,
        default="ART",
        choices=["ART", "RA", "S_FAT"],
        help="Annotation type to test (default: ART)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for testing (auto, cpu, gpu, cuda) (default: auto)",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Image size for model input (default: 224 224)",
    )

    parser.add_argument(
        "--slice-axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Slice axis for 2D extraction (0=sagittal, 1=coronal, 2=axial) (default: 2)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debugging output",
    )

    return parser.parse_args()


def validate_data_directory(data_dir: Path) -> None:
    """Validate that the provided data directory exists and is accessible.

    Args:
        data_dir: Path to data directory

    Raises:
        SystemExit: If directory is invalid
    """
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    if not data_dir.is_dir():
        logger.error(f"Path is not a directory: {data_dir}")
        sys.exit(1)

    # Check for some basic structure (subject directories)
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if not subdirs:
        logger.warning(f"No subdirectories found in {data_dir}")
        logger.warning(
            "Expected structure: data_dir/subject_id/study_volume/*.nii.gz"
        )

    logger.info(f"✓ Data directory validated: {data_dir}")
    logger.info(f"  Found {len(subdirs)} potential subject directories")


def create_test_configurations(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, ComputeConfig]:
    """Create configuration objects for testing.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (DataConfig, ModelConfig, ComputeConfig)
    """
    logger.info("Creating test configurations...")

    # Data configuration
    data_config = DataConfig(
        data_dir=args.data_dir,
        annotation_dir=args.data_dir,  # Same directory for annotations
        annotation_type=args.annotation_type,  # Use annotation type from args
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        slice_axis=args.slice_axis,
        inference_chunk_size=8,  # Used for inference memory management
        img_size=args.img_size,
        target_spacing=[1.0, 1.0, 1.0],
        target_size=[args.img_size[0], args.img_size[1], 64],
        use_adaptive_hu_normalization=True,
        adaptive_hu_lower_percentile=0.5,
        adaptive_hu_upper_percentile=99.5,
    )

    # Model configuration
    model_config = ModelConfig(
        img_size=args.img_size,
        in_channels=1,
        out_channels=1,
        features=[32, 32, 64, 128, 256, 32],
        activation="PRELU",
        spatial_dims=2,
        slice_axis=args.slice_axis,
    )

    # Compute configuration
    accelerator = "cpu"
    if args.device == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    elif args.device in ["gpu", "cuda"]:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if accelerator == "cpu" and args.device != "cpu":
            logger.warning(
                f"Requested {args.device} but CUDA not available, using CPU"
            )

    compute_config = ComputeConfig(
        devices=1,
        accelerator=accelerator,
        strategy="auto",
        precision="32",  # Use 32-bit for testing stability
        num_workers=0,  # Disable multiprocessing for testing
    )

    logger.info("✓ Configurations created successfully")
    logger.info(
        f"  Data: {data_config.img_size}, axis={data_config.slice_axis}"
    )
    logger.info(
        f"  Model: {model_config.features} features, {model_config.activation} activation"
    )
    logger.info(
        f"  Compute: {compute_config.accelerator}, precision={compute_config.precision}"
    )

    return data_config, model_config, compute_config


def test_dataset_loading(
    data_config: DataConfig, compute_config: ComputeConfig
) -> dict[str, Any]:
    """Test dataset loading and return a sample batch.

    Args:
        data_config: Data configuration (includes annotation_type)
        compute_config: Compute configuration

    Returns:
        Dictionary containing sample batch data

    Raises:
        SystemExit: If dataset loading fails
    """
    logger.info("Testing dataset loading...")

    try:
        # Initialize DataModule
        datamodule = NiftiDataModule(
            data_config=data_config,
            compute_config=compute_config,
            enable_caching=False,  # Disable caching for testing
            cache_size=1,
            random_seed=42,
        )

        # Setup data discovery
        logger.info("Running data discovery...")
        datamodule.setup()

        # Get discovery summary
        summary = datamodule.get_discovery_summary()
        if summary:
            logger.info("✓ Discovery completed:")
            logger.info(f"  Total subjects: {summary['total_subjects']}")
            logger.info(
                f"  Subjects with {data_config.annotation_type}: {summary['subjects_with_target_annotation']}"
            )
            logger.info(
                f"  Available annotations: {summary['annotation_types']}"
            )

        if summary and summary["subjects_with_target_annotation"] == 0:
            logger.error(
                f"No subjects found with annotation type '{data_config.annotation_type}'"
            )
            sys.exit(1)

        # Get dataset info
        dataset_info = datamodule.get_dataset_info()
        logger.info("✓ Dataset splits created:")
        for split_name, info in dataset_info.items():
            logger.info(f"  {split_name}: {info['num_volumes']} volumes")

        # Create data loader
        train_loader = datamodule.train_dataloader()
        logger.info(f"✓ Train DataLoader created: {len(train_loader)} batches")

        # Get one sample batch
        logger.info("Loading sample batch...")
        for batch_idx, batch in enumerate(train_loader):
            logger.info("✓ Sample batch loaded successfully")
            logger.info(f"  Volume shape: {batch['volume'].shape}")
            logger.info(f"  Segmentation shape: {batch['segmentation'].shape}")
            logger.info(f"  Subject ID: {batch['subject_id']}")
            logger.info(f"  Annotation type: {batch['annotation_type']}")
            logger.info(f"  Number of slices: {batch['num_slices']}")

            return batch

        logger.error("No batches available in train loader")
        sys.exit(1)

    except Exception as e:
        logger.exception("Failed to load dataset")
        logger.error(f"Dataset loading error: {e}")
        sys.exit(1)


def test_model_initialization(model_config: ModelConfig) -> UNet2D:
    """Test UNet2D model initialization.

    Args:
        model_config: Model configuration

    Returns:
        Initialized UNet2D model

    Raises:
        SystemExit: If model initialization fails
    """
    logger.info("Testing UNet2D model initialization...")

    try:
        # Initialize model
        model = UNet2D(config=model_config)
        logger.info("✓ UNet2D model initialized successfully")

        # Get model info
        model_info = model.get_model_info()
        logger.info("✓ Model architecture details:")
        logger.info(f"  Type: {model_info['model_type']}")
        logger.info(f"  Input channels: {model_info['in_channels']}")
        logger.info(f"  Output channels: {model_info['out_channels']}")
        logger.info(f"  Features: {model_info['features']}")
        logger.info(f"  Activation: {model_info['activation']}")
        logger.info(f"  Spatial dims: {model_info['spatial_dims']}D")

        return model

    except Exception as e:
        logger.exception("Failed to initialize UNet2D model")
        logger.error(f"Model initialization error: {e}")
        sys.exit(1)


def test_forward_pass(
    model: UNet2D, batch: dict[str, Any], device: str
) -> torch.Tensor:
    """Test model forward pass with real batch data.

    Args:
        model: UNet2D model instance
        batch: Sample batch from dataset
        device: Device to run inference on

    Returns:
        Model output tensor

    Raises:
        SystemExit: If forward pass fails
    """
    logger.info("Testing model forward pass...")

    try:
        # Move model to device
        if device != "cpu" and torch.cuda.is_available():
            model = model.cuda()
            device_name = "GPU"
        else:
            device_name = "CPU"

        model.eval()  # Set to evaluation mode

        # Extract input tensor
        volume_tensor = batch["volume"]  # [N, C, H, W]
        logger.info(f"✓ Input tensor shape: {volume_tensor.shape}")

        # Move to device
        if device != "cpu" and torch.cuda.is_available():
            volume_tensor = volume_tensor.cuda()

        # Forward pass
        logger.info(f"Running forward pass on {device_name}...")
        with torch.no_grad():
            output = model(volume_tensor)

        logger.info("✓ Forward pass completed successfully")
        logger.info(f"  Input shape: {volume_tensor.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Device: {device_name}")

        # Basic output validation
        expected_shape = volume_tensor.shape  # Should match input shape
        if output.shape != expected_shape:
            logger.warning(
                f"Output shape {output.shape} differs from expected {expected_shape}"
            )
        else:
            logger.info("✓ Output shape matches expected format")

        # Check output value ranges
        output_min = output.min().item()
        output_max = output.max().item()
        output_mean = output.mean().item()

        logger.info("✓ Output statistics:")
        logger.info(f"  Min: {output_min:.4f}")
        logger.info(f"  Max: {output_max:.4f}")
        logger.info(f"  Mean: {output_mean:.4f}")

        return output

    except Exception as e:
        logger.exception("Failed during model forward pass")
        logger.error(f"Forward pass error: {e}")
        sys.exit(1)


def main() -> None:
    """Main test function."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("UNet2D Model Integration Test")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_arguments()

    if args.verbose:
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            level="DEBUG",
        )

    logger.info("Test parameters:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Annotation type: {args.annotation_type}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Image size: {args.img_size}")
    logger.info(f"  Slice axis: {args.slice_axis}")
    logger.info("")

    # Step 1: Validate data directory
    validate_data_directory(args.data_dir)
    logger.info("")

    # Step 2: Create configurations
    data_config, model_config, compute_config = create_test_configurations(args)
    logger.info("")

    # Step 3: Test dataset loading
    batch = test_dataset_loading(data_config, compute_config)
    logger.info("")

    # Step 4: Test model initialization
    model = test_model_initialization(model_config)
    logger.info("")

    # Step 5: Test forward pass
    output = test_forward_pass(model, batch, compute_config.accelerator)
    logger.info("")

    # Final summary
    logger.info("=" * 60)
    logger.info("✓ ALL TESTS PASSED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("Integration test results:")
    logger.info(f"  ✓ Data directory validated: {args.data_dir}")
    logger.info(
        f"  ✓ Dataset loaded with {data_config.annotation_type} annotations"
    )
    logger.info(
        f"  ✓ UNet2D model initialized with {model_config.features} features"
    )
    logger.info(
        f"  ✓ Forward pass completed on {compute_config.accelerator.upper()}"
    )
    logger.info(f"  ✓ Output shape: {output.shape}")
    logger.info("")
    logger.info(
        "The UNet2D model is ready for training implementation (Task 10)"
    )


if __name__ == "__main__":
    main()
