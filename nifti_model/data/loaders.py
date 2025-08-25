"""Data loading and preprocessing for NIFTI volumes."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import lightning
import nibabel as nib
import numpy as np
import torch
from loguru import logger
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    Spacingd,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils.config import DataConfig


class NiftiVolumeDataset(Dataset):
    """Dataset for loading NIFTI volumes and annotations.
    
    This dataset loads 3D NIFTI volumes and corresponding segmentation masks.
    Each sample contains the full volume rather than slices, allowing for
    volume-based batch processing.
    """
    
    def __init__(
        self,
        data_dir: Path,
        annotation_dir: Optional[Path] = None,
        volume_pattern: str = "*.nii.gz",
        annotation_pattern: str = "*_seg.nii.gz",
        transform: Optional[Compose] = None,
    ) -> None:
        """Initialize NIFTI dataset.
        
        Args:
            data_dir: Directory containing NIFTI volume files
            annotation_dir: Directory containing annotation files (if separate)
            volume_pattern: Glob pattern for volume files
            annotation_pattern: Glob pattern for annotation files
            transform: MONAI transform pipeline
            
        Raises:
            ValueError: If no volumes found or volume/annotation count mismatch
        """
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else data_dir
        self.transform = transform
        
        # Find volume and annotation files
        self.volume_files = sorted(self.data_dir.glob(volume_pattern))
        self.annotation_files = sorted(self.annotation_dir.glob(annotation_pattern))
        
        if not self.volume_files:
            raise ValueError(f"No volumes found in {data_dir} with pattern {volume_pattern}")
        
        if self.annotation_files and len(self.volume_files) != len(self.annotation_files):
            logger.warning(
                f"Volume count ({len(self.volume_files)}) != "
                f"annotation count ({len(self.annotation_files)})"
            )
        
        # Create data dictionaries for MONAI
        self.data_dicts = []
        for i, volume_file in enumerate(self.volume_files):
            data_dict = {"image": str(volume_file)}
            
            if self.annotation_files:
                # Try to match annotation to volume by index or naming
                if i < len(self.annotation_files):
                    data_dict["label"] = str(self.annotation_files[i])
                else:
                    logger.warning(f"No annotation found for volume {volume_file}")
            
            self.data_dicts.append(data_dict)
        
        logger.info(f"Loaded {len(self.data_dicts)} volume-annotation pairs")
    
    def __len__(self) -> int:
        """Return number of volumes in dataset."""
        return len(self.data_dicts)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get volume and annotation at index.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dictionary containing image and optionally label tensors
        """
        data_dict = self.data_dicts[idx].copy()
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    @staticmethod
    def load_nifti_info(file_path: Path) -> dict[str, Any]:
        """Load NIFTI file metadata without loading full data.
        
        Args:
            file_path: Path to NIFTI file
            
        Returns:
            Dictionary with file info (shape, spacing, orientation, etc.)
        """
        try:
            nii = nib.load(str(file_path))
            header = nii.header
            
            return {
                "file_path": str(file_path),
                "shape": nii.shape,
                "voxel_size": header.get_zooms()[:3],
                "orientation": nib.aff2axcodes(nii.affine),
                "data_type": header.get_data_dtype(),
                "file_size_mb": file_path.stat().st_size / 1024 / 1024,
            }
        except Exception as e:
            logger.exception(f"Failed to load info for {file_path}")
            return {"file_path": str(file_path), "error": str(e)}


class NiftiSliceDataset(Dataset):
    """Dataset for loading 2D slices from NIFTI volumes.
    
    This dataset extracts individual 2D slices from 3D NIFTI volumes and
    their corresponding segmentation masks. Each sample contains a single
    slice, but slices are grouped by volume for batch processing.
    """
    
    def __init__(
        self,
        data_dir: Path,
        annotation_dir: Optional[Path] = None,
        volume_pattern: str = "*.nii.gz",
        annotation_pattern: str = "*_seg.nii.gz",
        transform: Optional[Compose] = None,
        slice_axis: int = 2,  # 0=axial, 1=coronal, 2=sagittal
        volume_as_batch: bool = True,  # Group slices by volume in batches
    ) -> None:
        """Initialize NIFTI slice dataset.
        
        Args:
            data_dir: Directory containing NIFTI volume files
            annotation_dir: Directory containing annotation files (if separate)
            volume_pattern: Glob pattern for volume files
            annotation_pattern: Glob pattern for annotation files
            transform: MONAI transform pipeline
            slice_axis: Axis along which to extract slices
            volume_as_batch: If True, group slices by volume for batching
            
        Raises:
            ValueError: If no volumes found or volume/annotation count mismatch
        """
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else data_dir
        self.transform = transform
        self.slice_axis = slice_axis
        self.volume_as_batch = volume_as_batch
        
        # Find volume and annotation files
        self.volume_files = sorted(self.data_dir.glob(volume_pattern))
        self.annotation_files = sorted(self.annotation_dir.glob(annotation_pattern))
        
        if not self.volume_files:
            raise ValueError(f"No volumes found in {data_dir} with pattern {volume_pattern}")
        
        if self.annotation_files and len(self.volume_files) != len(self.annotation_files):
            logger.warning(
                f"Volume count ({len(self.volume_files)}) != "
                f"annotation count ({len(self.annotation_files)})"
            )
        
        # Create slice-level data mappings
        self.slice_data = []
        self.volume_slice_counts = []
        
        for vol_idx, volume_file in enumerate(self.volume_files):
            # Load volume to get slice count
            volume_nii = nib.load(str(volume_file))
            volume_shape = volume_nii.shape
            num_slices = volume_shape[self.slice_axis]
            
            self.volume_slice_counts.append(num_slices)
            
            # Create slice entries
            for slice_idx in range(num_slices):
                slice_data = {
                    "volume_idx": vol_idx,
                    "slice_idx": slice_idx,
                    "image_path": str(volume_file),
                }
                
                if self.annotation_files and vol_idx < len(self.annotation_files):
                    slice_data["label_path"] = str(self.annotation_files[vol_idx])
                
                self.slice_data.append(slice_data)
        
        logger.info(
            f"Loaded {len(self.slice_data)} slices from {len(self.volume_files)} volumes"
        )
    
    def __len__(self) -> int:
        """Return number of slices in dataset."""
        return len(self.slice_data)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get slice and annotation at index.
        
        Args:
            idx: Dataset index (slice index)
            
        Returns:
            Dictionary containing slice image and optionally label tensors
        """
        slice_info = self.slice_data[idx]
        
        # Load volume data
        volume_nii = nib.load(slice_info["image_path"])
        volume_data = volume_nii.get_fdata()
        
        # Extract specific slice
        if self.slice_axis == 0:  # Axial
            slice_data = volume_data[slice_info["slice_idx"], :, :]
        elif self.slice_axis == 1:  # Coronal
            slice_data = volume_data[:, slice_info["slice_idx"], :]
        else:  # Sagittal (axis=2)
            slice_data = volume_data[:, :, slice_info["slice_idx"]]
        
        # Add channel dimension and convert to tensor
        slice_tensor = torch.from_numpy(slice_data).float().unsqueeze(0)  # (1, H, W)
        
        # Create data dictionary
        data_dict = {
            "image": slice_tensor,
            "volume_idx": slice_info["volume_idx"],
            "slice_idx": slice_info["slice_idx"],
        }
        
        # Load annotation if available
        if "label_path" in slice_info:
            label_nii = nib.load(slice_info["label_path"])
            label_data = label_nii.get_fdata()
            
            # Extract corresponding label slice
            if self.slice_axis == 0:  # Axial
                label_slice = label_data[slice_info["slice_idx"], :, :]
            elif self.slice_axis == 1:  # Coronal
                label_slice = label_data[:, slice_info["slice_idx"], :]
            else:  # Sagittal (axis=2)
                label_slice = label_data[:, :, slice_info["slice_idx"]]
            
            label_tensor = torch.from_numpy(label_slice).float().unsqueeze(0)
            data_dict["label"] = label_tensor
        
        # Apply transforms if specified (note: these should be 2D transforms)
        if self.transform:
            # Convert tensors back to numpy for MONAI transforms
            data_dict["image"] = data_dict["image"].numpy()
            if "label" in data_dict:
                data_dict["label"] = data_dict["label"].numpy()
            
            # Apply transforms
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def get_volume_slice_range(self, volume_idx: int) -> tuple[int, int]:
        """Get the slice index range for a specific volume.
        
        Args:
            volume_idx: Index of the volume
            
        Returns:
            Tuple of (start_idx, end_idx) for slices belonging to the volume
        """
        start_idx = sum(self.volume_slice_counts[:volume_idx])
        end_idx = start_idx + self.volume_slice_counts[volume_idx]
        return start_idx, end_idx


class VolumeBasedSampler(Sampler):
    """Sampler that groups slices by volume for batch processing.
    
    This sampler ensures that each batch contains slices from the same volume,
    which is required for the training approach where we process one volume
    per batch but train on individual slices.
    """
    
    def __init__(
        self, 
        slice_dataset: NiftiSliceDataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        """Initialize volume-based sampler.
        
        Args:
            slice_dataset: The slice dataset to sample from
            batch_size: Number of slices per batch (from same volume)
            shuffle: Whether to shuffle volumes and slices
        """
        self.dataset = slice_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group slice indices by volume
        self.volume_to_slices = defaultdict(list)
        for idx, slice_info in enumerate(slice_dataset.slice_data):
            self.volume_to_slices[slice_info["volume_idx"]].append(idx)
        
        # Calculate total batches
        self.total_batches = 0
        for volume_slices in self.volume_to_slices.values():
            self.total_batches += (len(volume_slices) + batch_size - 1) // batch_size
    
    def __iter__(self):
        """Generate batches of slice indices grouped by volume."""
        volume_indices = list(self.volume_to_slices.keys())
        if self.shuffle:
            random.shuffle(volume_indices)
        
        for volume_idx in volume_indices:
            slice_indices = self.volume_to_slices[volume_idx].copy()
            if self.shuffle:
                random.shuffle(slice_indices)
            
            # Create batches from this volume's slices
            for i in range(0, len(slice_indices), self.batch_size):
                batch = slice_indices[i:i + self.batch_size]
                yield batch
    
    def __len__(self):
        """Return total number of batches."""
        return self.total_batches


def create_transforms_2d(
    config: DataConfig,
    is_training: bool = True,
    target_size: Optional[tuple[int, int]] = None,
) -> Compose:
    """Create MONAI transform pipeline for 2D slices.
    
    Args:
        config: Data configuration
        is_training: Whether to include training augmentations  
        target_size: Target size for slice resizing (H, W)
        
    Returns:
        MONAI Compose transform pipeline for 2D slices
    """
    # Note: For slice-based processing, we assume image and label are already
    # extracted as 2D arrays and converted to tensors in the dataset
    
    # Check if we have labels
    has_labels = hasattr(config, 'annotation_dir') and config.annotation_dir is not None
    keys_with_label = ["image", "label"] if has_labels else ["image"]
    
    if target_size is None:
        target_size = (224, 224)  # Default 2D target size
    
    # Base transforms for 2D slices  
    base_transforms = [
        EnsureChannelFirstd(keys=keys_with_label),
        # Resize slices to target size
        Resized(
            keys=keys_with_label,
            spatial_size=target_size,
            mode=("bilinear", "nearest") if has_labels else "bilinear",
        ),
        # Normalize intensities
        NormalizeIntensityd(keys=["image"], subtrahend=None, divisor=None, nonzero=False),
    ]
    
    # Training augmentations for 2D
    if is_training:
        augment_transforms = [
            # 2D spatial augmentations
            RandFlipd(
                keys=keys_with_label,
                spatial_axis=[0, 1],  # Only H and W axes for 2D
                prob=0.5,
            ),
            RandRotate90d(
                keys=keys_with_label,
                prob=0.5,
                max_k=3,
                spatial_axes=(0, 1),  # Only rotate in H-W plane
            ),
            # Intensity augmentations
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            # Affine transformations for 2D
            RandAffined(
                keys=keys_with_label,
                mode=("bilinear", "nearest") if has_labels else "bilinear",
                prob=0.7,
                spatial_size=target_size,
                rotate_range=np.pi / 12,  # Single rotation angle for 2D
                scale_range=(0.1, 0.1),   # Only H, W scaling
                translate_range=(10, 10), # Only H, W translation
            ),
        ]
        
        base_transforms.extend(augment_transforms)
    
    # Final conversion to tensor
    base_transforms.append(
        ToTensord(keys=keys_with_label)
    )
    
    return Compose(base_transforms)


def create_transforms(
    config: DataConfig,
    is_training: bool = True,
    cache_transforms: bool = True,
) -> Compose:
    """Create MONAI transform pipeline (legacy 3D version).
    
    Args:
        config: Data configuration
        is_training: Whether to include training augmentations
        cache_transforms: Whether to cache deterministic transforms
        
    Returns:
        MONAI Compose transform pipeline
    """
    # This function is kept for backward compatibility but should
    # use create_transforms_2d for slice-based training
    logger.warning(
        "Using legacy 3D transforms. Consider using create_transforms_2d "
        "for slice-based training."
    )
    
    # Check if we have labels
    has_labels = hasattr(config, 'annotation_dir') and config.annotation_dir is not None
    keys_with_label = ["image", "label"] if has_labels else ["image"]
    
    # Base transforms (always applied)
    base_transforms = [
        LoadImaged(keys=keys_with_label),
        EnsureChannelFirstd(keys=keys_with_label),
        Orientationd(keys=keys_with_label, axcodes="RAS"),
    ]
    
    # Spacing normalization if specified
    if config.target_spacing:
        base_transforms.append(
            Spacingd(
                keys=keys_with_label,
                pixdim=config.target_spacing,
                mode=("bilinear", "nearest") if has_labels else "bilinear",
            )
        )
    
    # Note: Intensity normalization is handled in preprocessing pipeline
    # with proper Hounsfield Unit transforms
    base_transforms.extend([
        CropForegroundd(keys=keys_with_label),
    ])
    
    # Training augmentations
    if is_training:
        augment_transforms = [
            RandFlipd(
                keys=keys_with_label,
                spatial_axis=[0, 1, 2],
                prob=0.1,
            ),
            RandRotate90d(
                keys=keys_with_label,
                prob=0.1,
                max_k=3,
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            RandAffined(
                keys=keys_with_label,
                mode=("bilinear", "nearest") if has_labels else "bilinear",
                prob=1.0,
                spatial_size=config.target_size or [224, 224, 32],
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]
        
        base_transforms.extend(augment_transforms)
    
    # Final conversion to tensor
    base_transforms.append(
        ToTensord(keys=keys_with_label)
    )
    
    return Compose(base_transforms)


class NiftiDataModule(lightning.LightningDataModule):
    """Lightning DataModule for NIFTI slice data.
    
    Handles data loading, splitting, and transform creation for training,
    validation, and testing. Uses slice-based processing where each batch
    contains slices from the same volume.
    """
    
    def __init__(self, config: DataConfig) -> None:
        """Initialize data module.
        
        Args:
            config: Data configuration containing paths and parameters
        """
        super().__init__()
        self.config = config
        self.train_dataset: Optional[NiftiSliceDataset] = None
        self.val_dataset: Optional[NiftiSliceDataset] = None
        self.test_dataset: Optional[NiftiSliceDataset] = None
        
        # Store transforms
        self.train_transforms: Optional[Compose] = None
        self.val_transforms: Optional[Compose] = None
        
        # Add slice processing configuration
        self.slice_axis = getattr(config, 'slice_axis', 2)  # Default to sagittal
        self.batch_size = getattr(config, 'batch_size', 4)  # Slices per batch
    
    def prepare_data(self) -> None:
        """Prepare data (download, validate, etc.).
        
        This method is called only once and should contain data operations
        that should not be done on multiple processes.
        """
        # Validate data directory exists
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.config.data_dir}")
        
        # Log dataset statistics using slice dataset
        dataset = NiftiSliceDataset(
            data_dir=self.config.data_dir,
            annotation_dir=self.config.annotation_dir,
            volume_pattern=self.config.volume_pattern,
            annotation_pattern=self.config.annotation_pattern,
            slice_axis=self.slice_axis,
        )
        
        logger.info(f"Found {len(dataset)} slices in {self.config.data_dir}")
        
        # TODO: Add data validation (check file integrity, etc.)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training, validation, and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if stage == "fit" or stage is None:
            # Create full slice dataset
            full_dataset = NiftiSliceDataset(
                data_dir=self.config.data_dir,
                annotation_dir=self.config.annotation_dir,
                volume_pattern=self.config.volume_pattern,
                annotation_pattern=self.config.annotation_pattern,
                slice_axis=self.slice_axis,
            )
            
            # Split data by volumes (not by slices to keep volumes together)
            num_volumes = len(full_dataset.volume_files)
            train_vol_size = int(self.config.train_split * num_volumes)
            val_vol_size = int(self.config.val_split * num_volumes)
            
            # Get slice indices for train/val volumes
            train_slice_indices = []
            val_slice_indices = []
            
            for vol_idx in range(num_volumes):
                start_idx, end_idx = full_dataset.get_volume_slice_range(vol_idx)
                if vol_idx < train_vol_size:
                    train_slice_indices.extend(range(start_idx, end_idx))
                elif vol_idx < train_vol_size + val_vol_size:
                    val_slice_indices.extend(range(start_idx, end_idx))
            
            # Create transforms
            target_size = tuple(getattr(self.config, 'img_size', [224, 224])[:2])
            self.train_transforms = create_transforms_2d(
                self.config, is_training=True, target_size=target_size
            )
            self.val_transforms = create_transforms_2d(
                self.config, is_training=False, target_size=target_size
            )
            
            # Create subset datasets
            from torch.utils.data import Subset
            self.train_dataset = Subset(full_dataset, train_slice_indices)
            self.val_dataset = Subset(full_dataset, val_slice_indices)
            
            # Apply transforms (need to wrap datasets)
            self.train_dataset.dataset.transform = self.train_transforms
            self.val_dataset.dataset.transform = self.val_transforms
            
            logger.info(
                f"Created datasets: train={len(train_slice_indices)} slices, "
                f"val={len(val_slice_indices)} slices from {num_volumes} volumes"
            )
        
        if stage == "test":
            # TODO: Implement test dataset creation
            pass
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader with volume-based sampling.
        
        Returns:
            DataLoader for training data with volume-grouped batches
        """
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        # Create volume-based sampler for training
        # Note: We need access to the underlying NiftiSliceDataset
        slice_dataset = (self.train_dataset.dataset 
                        if hasattr(self.train_dataset, 'dataset') 
                        else self.train_dataset)
        
        sampler = VolumeBasedSampler(
            slice_dataset=slice_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader with volume-based sampling.
        
        Returns:
            DataLoader for validation data with volume-grouped batches
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        
        # Create volume-based sampler for validation
        slice_dataset = (self.val_dataset.dataset 
                        if hasattr(self.val_dataset, 'dataset') 
                        else self.val_dataset)
        
        sampler = VolumeBasedSampler(
            slice_dataset=slice_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation
        )
        
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader.
        
        Returns:
            DataLoader for test data
        """
        # TODO: Implement test data loader
        raise NotImplementedError("Test dataloader not implemented yet")
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader.
        
        Returns:
            DataLoader for prediction data
        """
        # TODO: Implement prediction data loader
        raise NotImplementedError("Prediction dataloader not implemented yet")