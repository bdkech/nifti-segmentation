#!/usr/bin/env python3
"""Example usage of NiftiDataModule for NIFTI volume processing.

This demonstrates how to use the NiftiDataModule for PyTorch Lightning training
with volume-based batching for medical image segmentation.
"""

from pathlib import Path

from niftilearn.config.models import ComputeConfig, DataConfig
from niftilearn.data import NiftiDataModule


def main():
    """Main example function demonstrating NiftiDataModule usage."""

    # Example configuration
    data_config = DataConfig(
        data_dir=Path("/path/to/nifti/data"),
        annotation_dir=Path(
            "/path/to/annotations"
        ),  # Not used in current implementation
        annotation_type="ART",  # Target artery annotations
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        slice_axis=2,  # Axial slices
        inference_chunk_size=8,  # Used for inference memory management, not training batching
        img_size=[224, 224],
        target_spacing=[1.0, 1.0, 1.0],
        target_size=[224, 224, 64],
    )

    compute_config = ComputeConfig(
        devices="auto",
        accelerator="gpu",
        strategy="ddp",
        precision="16-mixed",
        num_workers=4,
    )

    # Initialize DataModule
    datamodule = NiftiDataModule(
        data_config=data_config,
        compute_config=compute_config,
        enable_caching=True,  # Cache volumes for faster loading
        cache_size=10,  # Cache up to 10 volumes per dataset split
        random_seed=42,  # For reproducible splits
    )

    # Setup data discovery and dataset creation
    print("Setting up data discovery...")
    datamodule.setup()

    # Get discovery summary
    summary = datamodule.get_discovery_summary()
    if summary:
        print("Data Discovery Summary:")
        print(f"  Total subjects: {summary['total_subjects']}")
        print(
            f"  Subjects with {datamodule.annotation_type}: {summary['subjects_with_target_annotation']}"
        )
        print(f"  Available annotation types: {summary['annotation_types']}")

    # Get dataset information
    dataset_info = datamodule.get_dataset_info()
    print("\nDataset Information:")
    for split_name, info in dataset_info.items():
        print(f"  {split_name}:")
        print(f"    Volumes: {info['num_volumes']}")
        print(f"    Caching: {info['cache_enabled']}")
        print(
            f"    Cache utilization: {info['cache_stats']['utilization']:.1%}"
        )

    # Create data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print("\nDataLoaders Created:")
    print(f"  Train batches (volumes): {len(train_loader)}")
    print(f"  Val batches (volumes): {len(val_loader)}")
    print(f"  Test batches (volumes): {len(test_loader) if test_loader else 0}")

    # Example: Iterate through one training batch
    print("\nExample Training Batch:")
    for batch_idx, batch in enumerate(train_loader):
        print(f"  Batch {batch_idx}:")
        print(
            f"    Volume shape: {batch['volume'].shape}"
        )  # [N_slices, C, H, W]
        print(f"    Segmentation shape: {batch['segmentation'].shape}")
        print(f"    Subject ID: {batch['subject_id']}")
        print(f"    Annotation type: {batch['annotation_type']}")
        print(f"    Number of slices: {batch['num_slices']}")

        # Only show first batch for demonstration
        break

    # Validate configuration
    validation = datamodule.validate_configuration()
    print("\nConfiguration Validation:")
    print(f"  Valid: {validation['valid']}")
    if validation["issues"]:
        print(f"  Issues: {validation['issues']}")
    if validation["recommendations"]:
        print(f"  Recommendations: {validation['recommendations']}")


if __name__ == "__main__":
    main()
