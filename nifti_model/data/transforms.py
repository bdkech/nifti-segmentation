"""Advanced preprocessing and transforms pipeline for NIFTI volumes."""

from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import torch
from loguru import logger
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandShiftIntensityd,
    RandZoomd,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
)

from ..utils.config import DataConfig


class VolumeInfoLoggingd(MapTransform):
    """Custom transform to log volume information during preprocessing.
    
    This transform logs key volume statistics and properties to help
    understand the data distribution and preprocessing effects.
    """
    
    def __init__(
        self,
        keys: list[str],
        log_level: str = "DEBUG",
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize volume info logging transform.
        
        Args:
            keys: Keys to log information for
            log_level: Logging level for output
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.log_level = log_level
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Log volume information for specified keys.
        
        Args:
            data: Data dictionary
            
        Returns:
            Unchanged data dictionary
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            volume = d[key]
            
            if isinstance(volume, (torch.Tensor, np.ndarray, MetaTensor)):
                shape = volume.shape
                dtype = volume.dtype
                
                if hasattr(volume, "min") and hasattr(volume, "max"):
                    min_val = float(volume.min())
                    max_val = float(volume.max())
                    mean_val = float(volume.mean()) if hasattr(volume, "mean") else "N/A"
                else:
                    min_val = max_val = mean_val = "N/A"
                
                info = (
                    f"Volume {key}: shape={shape}, dtype={dtype}, "
                    f"range=[{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}"
                )
                
                if self.log_level.upper() == "DEBUG":
                    logger.debug(info)
                elif self.log_level.upper() == "INFO":
                    logger.info(info)
        
        return d


class HounsfieldUnitNormalizationd(MapTransform):
    """Custom transform for Hounsfield Unit (HU) intensity normalization.
    
    This transform handles medical imaging data with Hounsfield Units,
    applying clinically relevant intensity windowing and normalization.
    
    Common HU ranges:
    - Air: -1000 HU
    - Fat: -100 to -50 HU  
    - Water: 0 HU
    - Soft tissue: 10 to 40 HU
    - Muscle: 10 to 40 HU
    - Bone: 700 to 3000 HU
    """
    
    def __init__(
        self,
        keys: list[str],
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
        window_preset: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize Hounsfield Unit normalization.
        
        Args:
            keys: Keys to apply normalization to
            hu_min: Minimum HU value for windowing
            hu_max: Maximum HU value for windowing
            b_min: Target minimum value after normalization
            b_max: Target maximum value after normalization
            clip: Whether to clip values outside HU range
            window_preset: Predefined windowing preset ('soft_tissue', 'lung', 
                         'bone', 'brain', 'mediastinum', 'abdomen')
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        
        # Apply window presets if specified
        if window_preset:
            hu_min, hu_max = self._get_window_preset(window_preset)
        
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.window_preset = window_preset
        
        logger.debug(f"HU normalization: [{hu_min}, {hu_max}] -> [{b_min}, {b_max}]")
    
    def _get_window_preset(self, preset: str) -> tuple[float, float]:
        """Get predefined HU windowing ranges.
        
        Args:
            preset: Window preset name
            
        Returns:
            Tuple of (hu_min, hu_max) values
            
        Raises:
            ValueError: If preset is not recognized
        """
        presets = {
            "soft_tissue": (-160, 240),    # Soft tissue window
            "lung": (-1000, -300),         # Lung window
            "bone": (400, 1800),           # Bone window  
            "brain": (0, 80),              # Brain window
            "mediastinum": (-150, 250),    # Mediastinum window
            "abdomen": (-150, 250),        # Abdominal window
            "liver": (-20, 180),           # Liver window
            "kidney": (-30, 130),          # Kidney window
            "muscle": (-50, 150),          # Muscle window
            "cardiac": (-100, 200),        # Cardiac window
        }
        
        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown window preset: {preset}. Available: {available}")
        
        return presets[preset]
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply Hounsfield Unit normalization.
        
        Args:
            data: Data dictionary
            
        Returns:
            Data dictionary with HU-normalized volumes
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            volume = d[key]
            
            # Log original HU statistics
            logger.debug(
                f"Original HU range for {key}: "
                f"[{volume.min():.1f}, {volume.max():.1f}] HU"
            )
            
            # Apply HU windowing and normalization
            if self.hu_max > self.hu_min:
                # Clip to HU window if specified
                if self.clip:
                    volume_windowed = np.clip(volume, self.hu_min, self.hu_max)
                else:
                    volume_windowed = volume.copy()
                
                # Normalize to target range
                volume_norm = (volume_windowed - self.hu_min) / (self.hu_max - self.hu_min)
                volume_norm = volume_norm * (self.b_max - self.b_min) + self.b_min
                
                # Ensure output is in target range
                if self.clip:
                    volume_norm = np.clip(volume_norm, self.b_min, self.b_max)
                
                d[key] = volume_norm.astype(volume.dtype)
                
                logger.debug(
                    f"Normalized range for {key}: "
                    f"[{volume_norm.min():.3f}, {volume_norm.max():.3f}]"
                )
            else:
                logger.warning(f"Invalid HU range for {key}, skipping normalization")
        
        return d


class AdaptiveHUNormalizationd(MapTransform):
    """Adaptive Hounsfield Unit normalization using percentiles.
    
    This transform automatically determines appropriate HU windowing
    based on the data distribution, making it robust across different
    imaging protocols and anatomical regions.
    """
    
    def __init__(
        self,
        keys: list[str],
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        hu_min_limit: float = -1024.0,
        hu_max_limit: float = 3071.0,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize adaptive HU normalization.
        
        Args:
            keys: Keys to apply normalization to
            lower_percentile: Lower percentile for automatic windowing
            upper_percentile: Upper percentile for automatic windowing
            hu_min_limit: Absolute minimum HU value (CT scanner limit)
            hu_max_limit: Absolute maximum HU value (CT scanner limit)
            b_min: Target minimum value after normalization
            b_max: Target maximum value after normalization
            clip: Whether to clip values outside computed range
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.hu_min_limit = hu_min_limit
        self.hu_max_limit = hu_max_limit
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply adaptive HU normalization.
        
        Args:
            data: Data dictionary
            
        Returns:
            Data dictionary with adaptively normalized volumes
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            volume = d[key]
            
            # Calculate adaptive HU window based on percentiles
            hu_min = np.percentile(volume, self.lower_percentile)
            hu_max = np.percentile(volume, self.upper_percentile)
            
            # Ensure windows are within scanner limits
            hu_min = max(hu_min, self.hu_min_limit)
            hu_max = min(hu_max, self.hu_max_limit)
            
            logger.debug(
                f"Adaptive HU window for {key}: [{hu_min:.1f}, {hu_max:.1f}] HU "
                f"(percentiles {self.lower_percentile}-{self.upper_percentile})"
            )
            
            # Apply windowing and normalization
            if hu_max > hu_min:
                # Clip to computed window if specified
                if self.clip:
                    volume_windowed = np.clip(volume, hu_min, hu_max)
                else:
                    volume_windowed = volume.copy()
                
                # Normalize to target range
                volume_norm = (volume_windowed - hu_min) / (hu_max - hu_min)
                volume_norm = volume_norm * (self.b_max - self.b_min) + self.b_min
                
                # Ensure output is in target range
                if self.clip:
                    volume_norm = np.clip(volume_norm, self.b_min, self.b_max)
                
                d[key] = volume_norm.astype(volume.dtype)
            else:
                logger.warning(f"No HU range for {key}, skipping normalization")
        
        return d


class VolumeStatisticsLoggingd(MapTransform):
    """Transform to compute and log volume statistics for analysis."""
    
    def __init__(
        self,
        keys: list[str],
        compute_histogram: bool = False,
        num_bins: int = 100,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize volume statistics logging.
        
        Args:
            keys: Keys to compute statistics for
            compute_histogram: Whether to compute intensity histogram
            num_bins: Number of bins for histogram
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)
        self.compute_histogram = compute_histogram
        self.num_bins = num_bins
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute and log volume statistics.
        
        Args:
            data: Data dictionary
            
        Returns:
            Data dictionary with added statistics
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            volume = d[key]
            
            # Basic statistics
            stats = {
                "shape": volume.shape,
                "min": float(volume.min()),
                "max": float(volume.max()),
                "mean": float(volume.mean()),
                "std": float(volume.std()),
                "median": float(np.median(volume)),
            }
            
            # Percentile statistics
            percentiles = [1, 5, 25, 75, 95, 99]
            for p in percentiles:
                stats[f"p{p}"] = float(np.percentile(volume, p))
            
            # Non-zero statistics (for sparse volumes)
            nonzero_mask = volume != 0
            if nonzero_mask.any():
                nonzero_volume = volume[nonzero_mask]
                stats["nonzero_count"] = int(nonzero_mask.sum())
                stats["nonzero_fraction"] = float(nonzero_mask.mean())
                stats["nonzero_mean"] = float(nonzero_volume.mean())
                stats["nonzero_std"] = float(nonzero_volume.std())
            
            # Histogram
            if self.compute_histogram:
                hist, bin_edges = np.histogram(volume, bins=self.num_bins)
                stats["histogram"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            
            # Store statistics in data dictionary
            d[f"{key}_stats"] = stats
            
            logger.debug(f"Statistics for {key}: {stats}")
        
        return d


def create_preprocessing_pipeline(
    config: DataConfig,
    stage: str = "train",
    enable_augmentation: bool = True,
) -> Compose:
    """Create comprehensive preprocessing pipeline.
    
    Args:
        config: Data configuration
        stage: Processing stage ('train', 'val', 'test', 'predict')
        enable_augmentation: Whether to include data augmentation
        
    Returns:
        MONAI Compose transform pipeline
    """
    transforms = []
    
    # 1. Data Loading
    keys = ["image"]
    if stage in ["train", "val", "test"]:
        keys.append("label")
    
    transforms.extend([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        VolumeInfoLoggingd(keys=keys, log_level="DEBUG"),
    ])
    
    # 2. Orientation and Spacing Normalization
    transforms.extend([
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=config.target_spacing or (1.0, 1.0, 1.0),
            mode=("bilinear", "nearest") if "label" in keys else "bilinear",
        ) if config.target_spacing else lambda x: x,
    ])
    
    # Filter out no-op transforms
    transforms = [t for t in transforms if t is not None]
    
    # 3. Intensity Preprocessing (Hounsfield Unit handling)
    if stage != "predict":  # For prediction, assume input is already normalized
        # Choose HU normalization strategy based on config
        hu_window_preset = getattr(config, 'hu_window_preset', None)
        use_adaptive_hu = getattr(config, 'use_adaptive_hu_normalization', True)
        
        if use_adaptive_hu:
            # Adaptive HU normalization - automatically determines windows
            transforms.append(
                AdaptiveHUNormalizationd(
                    keys=["image"],
                    lower_percentile=getattr(config, 'adaptive_hu_lower_percentile', 0.5),
                    upper_percentile=getattr(config, 'adaptive_hu_upper_percentile', 99.5),
                    hu_min_limit=-1024.0,  # CT scanner minimum
                    hu_max_limit=3071.0,   # CT scanner maximum  
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )
            )
        else:
            # Fixed HU windowing - use preset or custom range
            hu_min = getattr(config, 'hu_min', -1000.0)
            hu_max = getattr(config, 'hu_max', 400.0)
            
            transforms.append(
                HounsfieldUnitNormalizationd(
                    keys=["image"],
                    hu_min=hu_min,
                    hu_max=hu_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                    window_preset=hu_window_preset,
                )
            )
    
    # 4. Spatial Preprocessing
    transforms.extend([
        CropForegroundd(
            keys=keys,
            source_key="image",
            margin=10,  # Add small margin around foreground
        ),
    ])
    
    # 5. Size Standardization
    if config.target_size:
        transforms.append(
            ResizeWithPadOrCropd(
                keys=keys,
                spatial_size=config.target_size,
                mode="constant",
                constant_values=0,
            )
        )
    
    # 6. Data Augmentation (training only)
    if enable_augmentation and stage == "train":
        # Spatial augmentations
        transforms.extend([
            # Random cropping with positive/negative sampling
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label" if "label" in keys else None,
                spatial_size=config.target_size or [224, 224, 32],
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ) if "label" in keys else None,
            
            # Geometric augmentations
            RandFlipd(
                keys=keys,
                spatial_axis=[0, 1, 2],
                prob=0.1,
            ),
            RandRotate90d(
                keys=keys,
                prob=0.1,
                max_k=3,
                spatial_axes=(0, 2),  # Rotate in axial plane
            ),
            RandRotated(
                keys=keys,
                range_x=np.pi / 36,  # ±5 degrees
                range_y=np.pi / 36,
                range_z=np.pi / 18,  # ±10 degrees
                prob=0.2,
                keep_size=True,
                mode=("bilinear", "nearest") if "label" in keys else "bilinear",
            ),
            RandZoomd(
                keys=keys,
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.2,
                mode=("bilinear", "nearest") if "label" in keys else "bilinear",
                align_corners=True,
            ),
            
            # Elastic deformation
            RandElasticd(
                keys=keys,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                spatial_size=config.target_size or [224, 224, 32],
                prob=0.1,
                rotate_range=np.pi / 36,
                scale_range=0.05,
                mode=("bilinear", "nearest") if "label" in keys else "bilinear",
            ),
        ])
        
        # Intensity augmentations (image only)
        transforms.extend([
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.5,
            ),
            RandGaussianNoised(
                keys=["image"],
                std=0.01,
                prob=0.15,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.25, 1.5),
                sigma_y=(0.25, 1.5),
                sigma_z=(0.25, 1.5),
                prob=0.1,
            ),
        ])
        
        # Filter out None transforms
        transforms = [t for t in transforms if t is not None]
    
    # 7. Final Processing
    transforms.extend([
        # Log final statistics
        VolumeStatisticsLoggingd(
            keys=keys,
            compute_histogram=False,  # Disable for performance
        ) if stage == "train" else lambda x: x,
        
        # Convert to tensor
        ToTensord(keys=keys),
    ])
    
    # Filter out no-op transforms
    transforms = [t for t in transforms if not callable(t) or t.__class__.__name__ != '<lambda>']
    
    logger.info(f"Created preprocessing pipeline with {len(transforms)} transforms for stage: {stage}")
    return Compose(transforms)


def create_inference_pipeline(
    target_spacing: Optional[tuple[float, ...]] = None,
    target_size: Optional[tuple[int, ...]] = None,
) -> Compose:
    """Create preprocessing pipeline for inference.
    
    Args:
        target_spacing: Target voxel spacing
        target_size: Target volume size
        
    Returns:
        MONAI Compose transform pipeline for inference
    """
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
    ]
    
    # Add spacing normalization if specified
    if target_spacing:
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=target_spacing,
                mode="bilinear",
            )
        )
    
    # Add size standardization if specified
    if target_size:
        transforms.append(
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=target_size,
                mode="constant",
                constant_values=0,
            )
        )
    
    # Final tensor conversion
    transforms.append(ToTensord(keys=["image"]))
    
    logger.info(f"Created inference pipeline with {len(transforms)} transforms")
    return Compose(transforms)


def preprocess_single_volume(
    volume_path: Path,
    output_path: Path,
    transforms: Compose,
    save_metadata: bool = True,
) -> dict[str, Any]:
    """Preprocess a single NIFTI volume.
    
    Args:
        volume_path: Path to input NIFTI volume
        output_path: Path to save preprocessed volume
        transforms: Transform pipeline to apply
        save_metadata: Whether to save preprocessing metadata
        
    Returns:
        Dictionary containing preprocessing metadata
        
    Raises:
        FileNotFoundError: If input volume doesn't exist
        Exception: If preprocessing fails
    """
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")
    
    logger.info(f"Preprocessing volume: {volume_path}")
    
    try:
        # Load and preprocess volume
        data = {"image": str(volume_path)}
        processed_data = transforms(data)
        
        # Extract processed volume
        processed_volume = processed_data["image"]
        
        # Save preprocessed volume
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(processed_volume, torch.Tensor):
            processed_volume = processed_volume.numpy()
        
        # Create NIFTI image
        nii_img = nib.Nifti1Image(
            processed_volume.squeeze(),  # Remove channel dimension for saving
            affine=np.eye(4),  # Identity affine for processed volumes
        )
        nib.save(nii_img, str(output_path))
        
        # Collect metadata
        metadata = {
            "input_path": str(volume_path),
            "output_path": str(output_path),
            "input_shape": processed_data.get("image_meta_dict", {}).get("spatial_shape", "unknown"),
            "output_shape": processed_volume.shape,
            "preprocessing_successful": True,
        }
        
        # Add statistics if available
        if "image_stats" in processed_data:
            metadata["statistics"] = processed_data["image_stats"]
        
        # Save metadata
        if save_metadata:
            metadata_path = output_path.with_suffix(".json")
            import json
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully preprocessed volume to: {output_path}")
        return metadata
        
    except Exception as e:
        logger.exception(f"Failed to preprocess volume {volume_path}")
        return {
            "input_path": str(volume_path),
            "preprocessing_successful": False,
            "error": str(e),
        }