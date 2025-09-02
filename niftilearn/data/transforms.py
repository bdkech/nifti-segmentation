"""2D slice extraction and volume preprocessing for NiftiLearn."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from niftilearn.config.models import DataConfig
from niftilearn.data.loaders import VolumeSegmentationPair


def get_hu_window_values(preset: str) -> tuple[float, float]:
    """Get HU window values for common presets.

    Args:
        preset: Window preset name ('soft_tissue', 'bone', 'lung')

    Returns:
        Tuple of (min_hu, max_hu) values
    """
    presets = {
        "muscle": (45, 50),
        "s_fat": (-100, -115),
        "arteries": (30, 60),
        "arteries_w_contrast": (250, 400),
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown HU preset: {preset}. Available: {list(presets.keys())}"
        )

    return presets[preset]


def create_volume_preprocessing_transform(data_config: DataConfig) -> Compose:
    """Create MONAI preprocessing transform for volumes.

    Args:
        data_config: Data configuration containing preprocessing parameters

    Returns:
        MONAI Compose transform for volume preprocessing
    """
    transforms = []

    # Load and ensure channel first
    transforms.extend(
        [
            LoadImaged(keys=["volume"], image_only=True),
            EnsureChannelFirstd(keys=["volume"]),
        ]
    )

    # Resampling to target spacing and size
    if data_config.target_spacing:
        transforms.append(
            Spacingd(
                keys=["volume"],
                pixdim=data_config.target_spacing,
                mode="bilinear",
            )
        )

    if data_config.target_size:
        transforms.append(
            ResizeWithPadOrCropd(
                keys=["volume"], spatial_size=data_config.target_size
            )
        )

    # Hounsfield Unit normalization
    if data_config.use_adaptive_hu_normalization:
        # For adaptive normalization, we'll use percentile-based scaling
        # This is a simplified approach - in practice you might want to compute percentiles first
        transforms.append(
            ScaleIntensityRanged(
                keys=["volume"],
                a_min=-1000,  # Typical HU range
                a_max=3000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        # Fixed HU windowing
        if data_config.hu_window_preset:
            hu_min, hu_max = get_hu_window_values(data_config.hu_window_preset)
        else:
            hu_min = (
                data_config.hu_min if data_config.hu_min is not None else -1000
            )
            hu_max = (
                data_config.hu_max if data_config.hu_max is not None else 3000
            )

        transforms.append(
            ScaleIntensityRanged(
                keys=["volume"],
                a_min=hu_min,
                a_max=hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )

    # Final normalization and type conversion
    transforms.extend(
        [
            EnsureTyped(keys=["volume"], dtype=torch.float32),
            ToTensord(keys=["volume"]),
        ]
    )

    return Compose(transforms)


def create_segmentation_preprocessing_transform(
    data_config: DataConfig,
) -> Compose:
    """Create MONAI preprocessing transform for segmentations.

    Args:
        data_config: Data configuration containing preprocessing parameters

    Returns:
        MONAI Compose transform for segmentation preprocessing
    """
    transforms = []

    # Load and ensure channel first
    transforms.extend(
        [
            LoadImaged(keys=["segmentation"], image_only=True),
            EnsureChannelFirstd(keys=["segmentation"]),
        ]
    )

    # Resampling to match volume processing
    if data_config.target_spacing:
        transforms.append(
            Spacingd(
                keys=["segmentation"],
                pixdim=data_config.target_spacing,
                mode="nearest",  # Use nearest for segmentation masks
            )
        )

    if data_config.target_size:
        transforms.append(
            ResizeWithPadOrCropd(
                keys=["segmentation"], spatial_size=data_config.target_size
            )
        )

    # Type conversion
    transforms.extend(
        [
            EnsureTyped(keys=["segmentation"], dtype=torch.float32),
            ToTensord(keys=["segmentation"]),
        ]
    )

    return Compose(transforms)


def preprocess_volume(
    volume_path: Path, data_config: DataConfig
) -> torch.Tensor:
    """Preprocess a single volume using MONAI transforms.

    Args:
        volume_path: Path to volume NIFTI file
        data_config: Data configuration

    Returns:
        Preprocessed volume tensor with shape [C, H, W, D]
    """
    transform = create_volume_preprocessing_transform(data_config)

    data = {"volume": str(volume_path)}
    result = transform(data)

    return result["volume"]


def preprocess_segmentation(
    segmentation_path: Path, data_config: DataConfig
) -> torch.Tensor:
    """Preprocess a single segmentation using MONAI transforms.

    Args:
        segmentation_path: Path to segmentation NIFTI file
        data_config: Data configuration

    Returns:
        Preprocessed segmentation tensor with shape [C, H, W, D]
    """
    transform = create_segmentation_preprocessing_transform(data_config)

    data = {"segmentation": str(segmentation_path)}
    result = transform(data)

    return result["segmentation"]


def extract_slices(
    volume_tensor: torch.Tensor,
    segmentation_tensor: torch.Tensor,
    slice_axis: int = 2,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract 2D slices from 3D volume and segmentation tensors.

    Args:
        volume_tensor: 3D volume tensor with shape [C, H, W, D]
        segmentation_tensor: 3D segmentation tensor with shape [C, H, W, D]
        slice_axis: Axis along which to extract slices (0, 1, or 2)

    Returns:
        List of (volume_slice, segmentation_slice) tuples, each with shape [C, H, W]
    """
    if volume_tensor.shape != segmentation_tensor.shape:
        raise ValueError(
            f"Volume and segmentation shapes must match: "
            f"{volume_tensor.shape} vs {segmentation_tensor.shape}"
        )

    # Adjust slice axis to account for channel dimension
    spatial_axis = slice_axis + 1  # +1 because of channel dimension at index 0

    slices = []
    num_slices = volume_tensor.shape[spatial_axis]

    for i in range(num_slices):
        if spatial_axis == 1:  # H axis
            vol_slice = volume_tensor[:, i, :, :]
            seg_slice = segmentation_tensor[:, i, :, :]
        elif spatial_axis == 2:  # W axis
            vol_slice = volume_tensor[:, :, i, :]
            seg_slice = segmentation_tensor[:, :, i, :]
        elif spatial_axis == 3:  # D axis (most common - axial slices)
            vol_slice = volume_tensor[:, :, :, i]
            seg_slice = segmentation_tensor[:, :, :, i]
        else:
            raise ValueError(
                f"Invalid slice axis: {slice_axis}. Must be 0, 1, or 2"
            )

        slices.append((vol_slice, seg_slice))

    return slices


class VolumeSliceExtractor:
    """Coordinates volume preprocessing and slice extraction for training."""

    def __init__(self, data_config: DataConfig):
        """Initialize with data configuration.

        Args:
            data_config: Data configuration containing processing parameters
        """
        self.data_config = data_config
        self.volume_transform = create_volume_preprocessing_transform(
            data_config
        )
        self.seg_transform = create_segmentation_preprocessing_transform(
            data_config
        )

        logger.info(
            f"VolumeSliceExtractor initialized with slice_axis={data_config.slice_axis}"
        )

    def process_volume_pair(
        self, pair: VolumeSegmentationPair, annotation_type: str
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Process a volume-segmentation pair into 2D slices.

        Args:
            pair: VolumeSegmentationPair containing volume and segmentation paths
            annotation_type: Type of annotation to use (e.g., 'ART', 'RA', 'S_FAT')

        Returns:
            List of (volume_slice, segmentation_slice) tuples for training

        Raises:
            ValueError: If annotation type not available for this pair
        """
        if not pair.has_annotation(annotation_type):
            raise ValueError(
                f"Annotation type '{annotation_type}' not available for subject {pair.subject_id}. "
                f"Available types: {pair.get_annotation_types()}"
            )

        segmentation_path = pair.get_segmentation_path(annotation_type)
        if segmentation_path is None:
            raise ValueError(f"Segmentation path is None for {annotation_type}")

        logger.debug(
            f"Processing {pair.subject_id} with annotation {annotation_type}"
        )

        try:
            # Preprocess volume
            volume_data = {"volume": str(pair.volume_path)}
            volume_result = self.volume_transform(volume_data)
            volume_tensor = volume_result["volume"]

            # Preprocess segmentation
            seg_data = {"segmentation": str(segmentation_path)}
            seg_result = self.seg_transform(seg_data)
            seg_tensor = seg_result["segmentation"]

            # Extract slices
            slices = extract_slices(
                volume_tensor,
                seg_tensor,
                slice_axis=self.data_config.slice_axis,
            )

            logger.debug(
                f"Extracted {len(slices)} slices from {pair.subject_id} "
                f"(annotation: {annotation_type})"
            )

            return slices

        except Exception:
            logger.exception(
                f"Failed to process {pair.subject_id} with annotation {annotation_type}"
            )
            raise

    def process_multiple_annotations(
        self, pair: VolumeSegmentationPair, annotation_types: list[str]
    ) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Process volume with multiple annotation types.

        Args:
            pair: VolumeSegmentationPair containing volume and segmentation paths
            annotation_types: List of annotation types to process

        Returns:
            Dictionary mapping annotation type to list of slice tuples
        """
        results = {}

        for annotation_type in annotation_types:
            if pair.has_annotation(annotation_type):
                try:
                    slices = self.process_volume_pair(pair, annotation_type)
                    results[annotation_type] = slices
                except Exception as e:
                    logger.error(
                        f"Failed to process {annotation_type} for {pair.subject_id}: {e}"
                    )
            else:
                logger.warning(
                    f"Annotation {annotation_type} not available for {pair.subject_id}"
                )

        return results

    def get_slice_shape(self) -> tuple[int, int]:
        """Get expected 2D slice shape after preprocessing.

        Returns:
            Tuple of (height, width) for 2D slices
        """
        target_size = self.data_config.target_size
        slice_axis = self.data_config.slice_axis

        # Remove the slicing dimension to get 2D shape
        spatial_dims = [target_size[i] for i in range(3) if i != slice_axis]
        return tuple(spatial_dims)
