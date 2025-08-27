"""NIFTI data discovery and loading utilities for NiftiLearn."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger

from niftilearn.config.models import DataConfig
from niftilearn.core.metadata import (
    extract_volume_metadata,
)


@dataclass
class VolumeSegmentationPair:
    """Represents a volume-segmentation pair for training."""

    subject_id: str
    volume_path: Path
    segmentation_paths: dict[str, Path]  # annotation_type -> path
    metadata: Optional[dict] = None

    def get_annotation_types(self) -> list[str]:
        """Return available annotation types (ART, RA, S_FAT).

        Returns:
            List of available annotation type strings
        """
        return list(self.segmentation_paths.keys())

    def has_annotation(self, annotation_type: str) -> bool:
        """Check if specific annotation type exists.

        Args:
            annotation_type: The annotation type to check (e.g., 'ART', 'RA', 'S_FAT')

        Returns:
            True if annotation type is available
        """
        return annotation_type in self.segmentation_paths

    def get_segmentation_path(self, annotation_type: str) -> Optional[Path]:
        """Get path for specific annotation type.

        Args:
            annotation_type: The annotation type to get path for

        Returns:
            Path to segmentation file or None if not found
        """
        return self.segmentation_paths.get(annotation_type)

    def validate_spatial_consistency(self) -> bool:
        """Validate that volume and segmentations have consistent spatial properties.

        Returns:
            True if spatially consistent
        """
        try:
            # Load volume metadata
            if self.metadata is None:
                volume_meta = extract_volume_metadata(self.volume_path)
            else:
                volume_meta = self.metadata

            volume_shape = volume_meta["shape"]
            volume_spacing = volume_meta["voxel_spacing"]

            # Check each segmentation
            for annotation_type, seg_path in self.segmentation_paths.items():
                seg_meta = extract_volume_metadata(seg_path)

                # Check shape consistency
                if seg_meta["shape"] != volume_shape:
                    logger.warning(
                        f"Shape mismatch for {self.subject_id} {annotation_type}: "
                        f"volume {volume_shape} vs segmentation {seg_meta['shape']}"
                    )
                    return False

                # Check spacing consistency (with tolerance)
                spacing_diff = np.array(seg_meta["voxel_spacing"]) - np.array(
                    volume_spacing
                )
                relative_diff = np.abs(spacing_diff) / np.array(volume_spacing)
                if np.any(relative_diff > 0.01):  # 1% tolerance
                    logger.warning(
                        f"Spacing mismatch for {self.subject_id} {annotation_type}: "
                        f"volume {volume_spacing} vs segmentation {seg_meta['voxel_spacing']}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(
                f"Failed to validate spatial consistency for {self.subject_id}: {e}"
            )
            return False


@dataclass
class DiscoveryResult:
    """Results from data discovery process."""

    pairs: list[VolumeSegmentationPair]
    subjects: list[str]
    annotation_types: set[str]
    discovery_stats: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def get_subjects_with_annotation(self, annotation_type: str) -> list[str]:
        """Get subjects that have a specific annotation type.

        Args:
            annotation_type: The annotation type to filter by

        Returns:
            List of subject IDs with the annotation type
        """
        return [
            pair.subject_id
            for pair in self.pairs
            if pair.has_annotation(annotation_type)
        ]

    def get_complete_subjects(
        self, required_annotations: list[str]
    ) -> list[str]:
        """Get subjects that have all required annotation types.

        Args:
            required_annotations: List of required annotation types

        Returns:
            List of subject IDs with all required annotations
        """
        complete_subjects = []
        for pair in self.pairs:
            if all(
                pair.has_annotation(ann_type)
                for ann_type in required_annotations
            ):
                complete_subjects.append(pair.subject_id)
        return complete_subjects

    def log_summary(self) -> None:
        """Log a summary of discovery results."""
        logger.info("Data Discovery Summary:")
        logger.info(f"  Total subjects: {len(self.subjects)}")
        logger.info(f"  Total volume-segmentation pairs: {len(self.pairs)}")
        logger.info(
            f"  Available annotation types: {sorted(self.annotation_types)}"
        )

        for ann_type in sorted(self.annotation_types):
            count = len(self.get_subjects_with_annotation(ann_type))
            logger.info(f"    {ann_type}: {count} subjects")

        if self.warnings:
            logger.warning(f"  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                logger.warning(f"    {warning}")
            if len(self.warnings) > 5:
                logger.warning(
                    f"    ... and {len(self.warnings) - 5} more warnings"
                )

        if self.errors:
            logger.error(f"  Errors: {len(self.errors)}")
            for error in self.errors[:3]:  # Show first 3 errors
                logger.error(f"    {error}")
            if len(self.errors) > 3:
                logger.error(f"    ... and {len(self.errors) - 3} more errors")


class NiftiDataDiscovery:
    """NIFTI data discovery and validation system."""

    def __init__(self, data_config: DataConfig):
        """Initialize with data configuration.

        Args:
            data_config: Data configuration containing paths and parameters
        """
        self.data_config = data_config
        self.expected_annotation_types = {"ART", "RA", "S_FAT"}

        # Validate data directory exists
        if not self.data_config.data_dir.exists():
            raise ValueError(
                f"Data directory does not exist: {self.data_config.data_dir}"
            )

        if not self.data_config.data_dir.is_dir():
            raise ValueError(
                f"Data path is not a directory: {self.data_config.data_dir}"
            )

    def discover_all_data(self) -> DiscoveryResult:
        """Main discovery method - scan entire data directory.

        Returns:
            DiscoveryResult containing all discovered volume-segmentation pairs
        """
        logger.info(f"Starting data discovery in: {self.data_config.data_dir}")

        result = DiscoveryResult(
            pairs=[],
            subjects=[],
            annotation_types=set(),
            discovery_stats={
                "total_subjects_found": 0,
                "valid_pairs_created": 0,
                "volumes_found": 0,
                "segmentations_found": 0,
            },
        )

        try:
            # Find all subject directories
            subject_dirs = [
                d
                for d in self.data_config.data_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

            result.discovery_stats["total_subjects_found"] = len(subject_dirs)
            logger.info(
                f"Found {len(subject_dirs)} potential subject directories"
            )

            # Process each subject
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name
                logger.debug(f"Processing subject: {subject_id}")

                try:
                    subject_pairs = self.discover_subject_data(subject_id)
                    result.pairs.extend(subject_pairs)

                    if subject_pairs:
                        result.subjects.append(subject_id)
                        # Collect annotation types
                        for pair in subject_pairs:
                            result.annotation_types.update(
                                pair.get_annotation_types()
                            )

                except Exception as e:
                    error_msg = f"Failed to process subject {subject_id}: {e}"
                    result.errors.append(error_msg)
                    logger.exception(f"Error processing subject {subject_id}")

            # Update final stats
            result.discovery_stats["valid_pairs_created"] = len(result.pairs)

            # Log summary
            result.log_summary()

            return result

        except Exception as e:
            logger.exception("Critical error during data discovery")
            result.errors.append(f"Critical discovery error: {e}")
            return result

    def discover_subject_data(
        self, subject_id: str
    ) -> list[VolumeSegmentationPair]:
        """Discover data for single subject.

        Args:
            subject_id: The subject identifier (directory name)

        Returns:
            List of VolumeSegmentationPair objects for the subject
        """
        subject_dir = self.data_config.data_dir / subject_id

        if not subject_dir.exists():
            raise ValueError(f"Subject directory does not exist: {subject_dir}")

        # Find volumes and segmentations
        volumes = self._find_volumes(subject_dir)
        segmentations = self._find_segmentations(subject_dir)

        logger.debug(
            f"Subject {subject_id}: found {len(volumes)} volumes, {len(segmentations)} segmentations"
        )

        if not volumes:
            logger.warning(f"No volumes found for subject {subject_id}")
            return []

        if not segmentations:
            logger.warning(f"No segmentations found for subject {subject_id}")
            return []

        # Create volume-segmentation pairs
        pairs = self._match_volumes_to_segmentations(
            subject_id, volumes, segmentations
        )

        # Validate each pair
        validated_pairs = []
        for pair in pairs:
            if self._validate_pair(pair):
                validated_pairs.append(pair)

        return validated_pairs

    def _find_volumes(self, subject_dir: Path) -> list[Path]:
        """Find all NIFTI volumes in subject subdirectories.

        Args:
            subject_dir: Path to subject directory

        Returns:
            List of paths to volume files
        """
        # Recursively find all .nii.gz files
        all_nifti = list(subject_dir.rglob("*.nii.gz"))

        # Filter out segmentation files
        volumes = [
            path
            for path in all_nifti
            if not path.name.startswith("Segmentations.")
        ]

        return sorted(volumes)

    def _find_segmentations(self, subject_dir: Path) -> dict[str, Path]:
        """Find segmentation files matching pattern Segmentations.*.nii.gz.

        Args:
            subject_dir: Path to subject directory

        Returns:
            Dictionary mapping annotation type to segmentation file path
        """
        segmentations = {}

        # Look for files matching the pattern
        pattern_files = list(subject_dir.glob("Segmentations.*.nii.gz"))

        for seg_path in pattern_files:
            # Parse annotation type from filename
            # Expected format: Segmentations.{annotation_type}.nii.gz
            name_parts = seg_path.name.split(".")

            if len(name_parts) >= 3 and name_parts[0] == "Segmentations":
                annotation_type = name_parts[1]

                # Validate annotation type
                if annotation_type in self.expected_annotation_types:
                    segmentations[annotation_type] = seg_path
                else:
                    logger.warning(
                        f"Unknown annotation type '{annotation_type}' in file: {seg_path.name}"
                    )

        return segmentations

    def _match_volumes_to_segmentations(
        self,
        subject_id: str,
        volumes: list[Path],
        segmentations: dict[str, Path],
    ) -> list[VolumeSegmentationPair]:
        """Create volume-segmentation pairs.

        Args:
            subject_id: The subject identifier
            volumes: List of volume file paths
            segmentations: Dictionary of annotation type to segmentation path

        Returns:
            List of VolumeSegmentationPair objects
        """
        pairs = []

        # For now, create one pair per volume with all available segmentations
        # In the future, we might implement more sophisticated matching logic
        for volume_path in volumes:
            pair = VolumeSegmentationPair(
                subject_id=subject_id,
                volume_path=volume_path,
                segmentation_paths=segmentations.copy(),
            )
            pairs.append(pair)

        return pairs

    def _validate_pair(self, pair: VolumeSegmentationPair) -> bool:
        """Validate volume-segmentation spatial consistency.

        Args:
            pair: VolumeSegmentationPair to validate

        Returns:
            True if pair is valid
        """
        try:
            # Check if files exist and are readable
            if not pair.volume_path.exists():
                logger.error(f"Volume file does not exist: {pair.volume_path}")
                return False

            for ann_type, seg_path in pair.segmentation_paths.items():
                if not seg_path.exists():
                    logger.error(
                        f"Segmentation file does not exist: {seg_path}"
                    )
                    return False

            # Extract volume metadata
            try:
                pair.metadata = extract_volume_metadata(pair.volume_path)
            except Exception as e:
                logger.error(
                    f"Failed to extract volume metadata for {pair.subject_id}: {e}"
                )
                return False

            # Validate spatial consistency
            if not pair.validate_spatial_consistency():
                return False

            logger.debug(
                f"Successfully validated pair for subject {pair.subject_id}"
            )
            return True

        except Exception:
            logger.exception(f"Validation failed for {pair.subject_id}")
            return False


def discover_nifti_data(data_config: DataConfig) -> DiscoveryResult:
    """Main function to discover NIFTI data.

    Args:
        data_config: Data configuration containing paths and parameters

    Returns:
        DiscoveryResult with all discovered and validated data
    """
    discovery = NiftiDataDiscovery(data_config)
    return discovery.discover_all_data()


def validate_data_directory(
    data_dir: Path,
) -> dict[str, Union[bool, list[str]]]:
    """Comprehensive data directory validation.

    Args:
        data_dir: Path to data directory to validate

    Returns:
        Dictionary containing validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    try:
        # Check directory exists
        if not data_dir.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Directory does not exist: {data_dir}"
            )
            return validation_result

        if not data_dir.is_dir():
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Path is not a directory: {data_dir}"
            )
            return validation_result

        # Check permissions
        if not os.access(data_dir, os.R_OK):
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Directory is not readable: {data_dir}"
            )
            return validation_result

        # Count subjects and files
        subject_dirs = [
            d
            for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        total_nifti = list(data_dir.rglob("*.nii.gz"))
        segmentation_files = [
            f for f in total_nifti if f.name.startswith("Segmentations.")
        ]
        volume_files = [
            f for f in total_nifti if not f.name.startswith("Segmentations.")
        ]

        validation_result["stats"] = {
            "subject_directories": len(subject_dirs),
            "total_nifti_files": len(total_nifti),
            "volume_files": len(volume_files),
            "segmentation_files": len(segmentation_files),
        }

        # Basic validation warnings
        if not subject_dirs:
            validation_result["warnings"].append("No subject directories found")

        if not volume_files:
            validation_result["warnings"].append("No volume files found")

        if not segmentation_files:
            validation_result["warnings"].append("No segmentation files found")

        logger.info(f"Directory validation completed for: {data_dir}")
        logger.info(
            f"  Found {len(subject_dirs)} subjects, {len(volume_files)} volumes, {len(segmentation_files)} segmentations"
        )

    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Validation failed: {e}")
        logger.exception(f"Directory validation failed for {data_dir}")

    return validation_result
