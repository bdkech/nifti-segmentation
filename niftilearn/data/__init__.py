"""Data loading and processing for NiftiLearn."""

# Core data structures
# PyTorch Lightning DataModule
from niftilearn.data.datamodule import NiftiDataModule

# PyTorch datasets
from niftilearn.data.datasets import (
    SliceDataItem,
    VolumeDataBatch,
    VolumeSliceDataset,
    create_datasets_from_discovery,
    create_volume_level_splits,
    validate_dataset_batch_size,
    volume_batch_collate,
)
from niftilearn.data.loaders import (
    DiscoveryResult,
    NiftiDataDiscovery,
    VolumeSegmentationPair,
    discover_nifti_data,
    validate_data_directory,
)

# Data transformations
from niftilearn.data.transforms import VolumeSliceExtractor

__all__ = [
    # Core data structures
    "DiscoveryResult",
    "NiftiDataDiscovery",
    # PyTorch Lightning DataModule
    "NiftiDataModule",
    # PyTorch datasets
    "SliceDataItem",
    "VolumeDataBatch",
    "VolumeSegmentationPair",
    "VolumeSliceDataset",
    # Data transformations
    "VolumeSliceExtractor",
    "create_datasets_from_discovery",
    "create_volume_level_splits",
    "discover_nifti_data",
    "validate_data_directory",
    "validate_dataset_batch_size",
    "volume_batch_collate",
]
