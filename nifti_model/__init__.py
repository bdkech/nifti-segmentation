"""NIFTI volume segmentation pipeline using 2D UNet slice-based architecture.

This package provides a complete pipeline for medical volume segmentation:
- NIFTI data loading with slice extraction and Hounsfield Unit support
- 2D UNet model architecture with Lightning integration  
- Slice-based training workflows with volume-level batching
- Training workflows with experiment tracking
- Volume reconstruction from slice predictions
"""

__version__ = "0.1.0"

# Import main components for easy access
from .data import NiftiDataModule, NiftiVolumeDataset, NiftiSliceDataset
from .models import UNet2DModel
from .training import TrainingManager
from .utils import Config, setup_logging

__all__ = [
    "Config",
    "NiftiDataModule",
    # Core components
    "NiftiVolumeDataset",
    "NiftiSliceDataset", 
    "TrainingManager",
    "UNet2DModel",
    "setup_logging",
]