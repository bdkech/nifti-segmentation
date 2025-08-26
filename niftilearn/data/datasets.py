"""PyTorch Dataset classes for slice-based NIFTI volume training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from torch.utils.data import Dataset

from niftilearn.config.models import DataConfig
from niftilearn.data.loaders import DiscoveryResult, VolumeSegmentationPair
from niftilearn.data.transforms import VolumeSliceExtractor


@dataclass
class SliceDataItem:
    """Container for individual slice data."""

    volume_slice: torch.Tensor  # [C, H, W]
    segmentation_slice: torch.Tensor  # [C, H, W]
    slice_index: int
    subject_id: str
    annotation_type: str


@dataclass
class VolumeDataBatch:
    """Container for volume-based batch data."""

    volume_slices: torch.Tensor  # [N, C, H, W] where N = number of slices
    segmentation_slices: torch.Tensor  # [N, C, H, W]
    slice_indices: list[int]
    subject_id: str
    annotation_type: str
    num_slices: int


def create_volume_level_splits(
    pairs: list[VolumeSegmentationPair],
    data_config: DataConfig,
    random_seed: int = 42,
) -> dict[str, list[VolumeSegmentationPair]]:
    """Create train/validation/test splits at volume level.

    Args:
        pairs: List of VolumeSegmentationPair objects
        data_config: Data configuration containing split ratios
        random_seed: Random seed for reproducible splits

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to volume lists

    Raises:
        ValueError: If splits result in empty sets or invalid ratios
    """
    if not pairs:
        raise ValueError("Cannot create splits from empty pairs list")

    # Set random seed for reproducible splits
    random.seed(random_seed)
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)

    total_volumes = len(pairs_copy)
    train_size = int(total_volumes * data_config.train_split)
    val_size = int(total_volumes * data_config.val_split)

    # Ensure we don't have empty splits
    if train_size == 0:
        raise ValueError(
            f"Train split too small: {data_config.train_split} of {total_volumes} volumes"
        )
    if val_size == 0:
        logger.warning(
            f"Validation split is empty: {data_config.val_split} of {total_volumes} volumes"
        )

    splits = {
        "train": pairs_copy[:train_size],
        "val": pairs_copy[train_size : train_size + val_size],
        "test": pairs_copy[train_size + val_size :],
    }

    # Log split statistics
    logger.info("Volume-level splits created:")
    logger.info(
        f"  Train: {len(splits['train'])} volumes ({len(splits['train']) / total_volumes:.2%})"
    )
    logger.info(
        f"  Val: {len(splits['val'])} volumes ({len(splits['val']) / total_volumes:.2%})"
    )
    logger.info(
        f"  Test: {len(splits['test'])} volumes ({len(splits['test']) / total_volumes:.2%})"
    )

    return splits


def volume_batch_collate(batch: list[VolumeDataBatch]) -> dict[str, Any]:
    """Custom collate function for volume-based batches.

    Args:
        batch: List containing single VolumeDataBatch (batch_size=1 for volumes)

    Returns:
        Dictionary with batched tensors and metadata

    Raises:
        ValueError: If batch contains more than one volume or is empty
    """
    if len(batch) != 1:
        raise ValueError(
            f"Volume batch collate expects exactly 1 volume per batch, got {len(batch)}. "
            f"Set DataLoader batch_size=1 for volume-based processing."
        )

    volume_batch = batch[0]

    return {
        "volume": volume_batch.volume_slices,  # [N, C, H, W]
        "segmentation": volume_batch.segmentation_slices,  # [N, C, H, W]
        "slice_indices": torch.tensor(
            volume_batch.slice_indices, dtype=torch.long
        ),
        "subject_id": volume_batch.subject_id,
        "annotation_type": volume_batch.annotation_type,
        "num_slices": volume_batch.num_slices,
    }


class VolumeSliceDataset(Dataset[VolumeDataBatch]):
    """PyTorch Dataset for slice-based volume training with 'one volume per batch' processing."""

    def __init__(
        self,
        pairs: list[VolumeSegmentationPair],
        annotation_type: str,
        data_config: DataConfig,
        split_type: str = "train",
        enable_caching: bool = False,
        cache_size: int = 10,
    ):
        """Initialize the volume slice dataset.

        Args:
            pairs: List of VolumeSegmentationPair objects for this split
            annotation_type: Target annotation type (e.g., 'ART', 'RA', 'S_FAT')
            data_config: Data configuration
            split_type: Split type ('train', 'val', 'test') for logging
            enable_caching: Whether to cache processed volumes
            cache_size: Maximum number of volumes to cache

        Raises:
            ValueError: If no pairs have the required annotation type
        """
        self.data_config = data_config
        self.annotation_type = annotation_type
        self.split_type = split_type
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        # Filter pairs that have the required annotation type
        self.pairs = [
            pair for pair in pairs if pair.has_annotation(annotation_type)
        ]

        if not self.pairs:
            raise ValueError(
                f"No volumes found with annotation type '{annotation_type}' "
                f"in {split_type} split"
            )

        # Initialize volume slice extractor
        self.extractor = VolumeSliceExtractor(data_config)

        # Initialize cache if enabled
        self.cache: dict[str, VolumeDataBatch] = {}
        self.cache_access_order: list[str] = []

        logger.info(
            f"VolumeSliceDataset created: {len(self.pairs)} volumes "
            f"({split_type}, annotation: {annotation_type})"
        )

        # Log expected slice shape
        slice_shape = self.extractor.get_slice_shape()
        logger.info(f"Expected slice shape: {slice_shape}")

    def __len__(self) -> int:
        """Return number of volumes (not slices)."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> VolumeDataBatch:
        """Get all slices from a volume as a batch.

        Args:
            idx: Volume index

        Returns:
            VolumeDataBatch containing all slices from the volume

        Raises:
            IndexError: If index is out of range
            RuntimeError: If volume processing fails
        """
        if idx >= len(self.pairs):
            raise IndexError(
                f"Index {idx} out of range for {len(self.pairs)} volumes"
            )

        pair = self.pairs[idx]
        cache_key = f"{pair.subject_id}_{self.annotation_type}"

        # Check cache first
        if self.enable_caching and cache_key in self.cache:
            # Update cache access order
            self.cache_access_order.remove(cache_key)
            self.cache_access_order.append(cache_key)
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]

        # Process volume
        try:
            slices = self.extractor.process_volume_pair(
                pair, self.annotation_type
            )

            if not slices:
                raise RuntimeError(f"No slices generated for {pair.subject_id}")

            # Convert to tensors
            volume_slices = torch.stack([vol_slice for vol_slice, _ in slices])
            segmentation_slices = torch.stack(
                [seg_slice for _, seg_slice in slices]
            )

            # Create batch object
            batch = VolumeDataBatch(
                volume_slices=volume_slices,
                segmentation_slices=segmentation_slices,
                slice_indices=list(range(len(slices))),
                subject_id=pair.subject_id,
                annotation_type=self.annotation_type,
                num_slices=len(slices),
            )

            # Cache if enabled
            if self.enable_caching:
                self._update_cache(cache_key, batch)

            logger.debug(
                f"Processed {pair.subject_id}: {batch.num_slices} slices "
                f"(shape: {batch.volume_slices.shape})"
            )

            return batch

        except Exception as e:
            logger.exception(f"Failed to process volume {pair.subject_id}")
            raise RuntimeError(
                f"Volume processing failed for {pair.subject_id}: {e}"
            ) from e

    def _update_cache(self, key: str, batch: VolumeDataBatch) -> None:
        """Update cache with LRU eviction.

        Args:
            key: Cache key
            batch: Volume batch to cache
        """
        # Remove oldest if cache is full
        if len(self.cache) >= self.cache_size and key not in self.cache:
            oldest_key = self.cache_access_order.pop(0)
            del self.cache[oldest_key]
            logger.debug(f"Cache evicted {oldest_key}")

        # Add/update cache
        self.cache[key] = batch
        if key in self.cache_access_order:
            self.cache_access_order.remove(key)
        self.cache_access_order.append(key)

        logger.debug(
            f"Cache updated for {key} ({len(self.cache)}/{self.cache_size})"
        )

    def get_volume_info(self, idx: int) -> dict[str, Any]:
        """Get metadata information for a volume without processing it.

        Args:
            idx: Volume index

        Returns:
            Dictionary with volume metadata
        """
        if idx >= len(self.pairs):
            raise IndexError(
                f"Index {idx} out of range for {len(self.pairs)} volumes"
            )

        pair = self.pairs[idx]
        return {
            "subject_id": pair.subject_id,
            "volume_path": str(pair.volume_path),
            "segmentation_path": str(
                pair.get_segmentation_path(self.annotation_type)
            ),
            "annotation_type": self.annotation_type,
            "available_annotations": pair.get_annotation_types(),
            "metadata": pair.metadata,
        }

    def clear_cache(self) -> None:
        """Clear the volume cache."""
        if self.enable_caching:
            self.cache.clear()
            self.cache_access_order.clear()
            logger.info("Volume cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self.enable_caching,
            "size": len(self.cache),
            "capacity": self.cache_size,
            "utilization": len(self.cache) / self.cache_size
            if self.cache_size > 0
            else 0,
            "cached_volumes": list(self.cache.keys()),
        }


def create_datasets_from_discovery(
    discovery_result: DiscoveryResult,
    annotation_type: str,
    data_config: DataConfig,
    random_seed: int = 42,
    enable_caching: bool = False,
    cache_size: int = 10,
) -> dict[str, VolumeSliceDataset]:
    """Create train/val/test datasets from discovery results.

    Args:
        discovery_result: Results from data discovery
        annotation_type: Target annotation type
        data_config: Data configuration
        random_seed: Random seed for splits
        enable_caching: Enable volume caching
        cache_size: Cache size for each dataset

    Returns:
        Dictionary with 'train', 'val', 'test' datasets

    Raises:
        ValueError: If no volumes available for annotation type
    """
    # Get volumes with required annotation type
    available_pairs = [
        pair
        for pair in discovery_result.pairs
        if pair.has_annotation(annotation_type)
    ]

    if not available_pairs:
        raise ValueError(
            f"No volumes found with annotation type '{annotation_type}'. "
            f"Available types: {discovery_result.annotation_types}"
        )

    logger.info(
        f"Creating datasets for annotation '{annotation_type}': "
        f"{len(available_pairs)} volumes available"
    )

    # Create volume-level splits
    splits = create_volume_level_splits(
        available_pairs, data_config, random_seed
    )

    # Create datasets for each split
    datasets = {}
    for split_name, split_pairs in splits.items():
        if split_pairs:  # Only create dataset if split is not empty
            datasets[split_name] = VolumeSliceDataset(
                pairs=split_pairs,
                annotation_type=annotation_type,
                data_config=data_config,
                split_type=split_name,
                enable_caching=enable_caching,
                cache_size=cache_size,
            )
        else:
            logger.warning(f"Empty {split_name} split - no dataset created")

    return datasets


def validate_dataset_batch_size(
    dataset: VolumeSliceDataset, target_batch_size: int
) -> bool:
    """Validate that dataset volumes produce expected batch sizes.

    Args:
        dataset: Dataset to validate
        target_batch_size: Expected batch size (number of slices per volume)

    Returns:
        True if all volumes produce target batch size
    """
    logger.info(f"Validating dataset batch sizes (target: {target_batch_size})")

    mismatched_volumes = []

    for idx in range(min(len(dataset), 10)):  # Check first 10 volumes
        try:
            batch = dataset[idx]
            if batch.num_slices != target_batch_size:
                mismatched_volumes.append(
                    {
                        "subject_id": batch.subject_id,
                        "expected": target_batch_size,
                        "actual": batch.num_slices,
                    }
                )
        except Exception as e:
            logger.error(f"Failed to validate volume {idx}: {e}")
            return False

    if mismatched_volumes:
        logger.warning(
            f"Found {len(mismatched_volumes)} volumes with mismatched batch sizes:"
        )
        for mismatch in mismatched_volumes:
            logger.warning(
                f"  {mismatch['subject_id']}: expected {mismatch['expected']}, "
                f"got {mismatch['actual']}"
            )
        return False

    logger.info("Dataset batch size validation passed")
    return True
