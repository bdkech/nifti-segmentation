"""Volume metadata utilities for NiftiLearn."""

from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from loguru import logger


def extract_volume_metadata(
    volume_path: Path,
) -> dict[str, Union[str, int, float, list, tuple]]:
    """Extract essential metadata from a NIFTI volume for 3D reconstruction.

    This function extracts only the metadata necessary for reconstructing 3D volumes
    from slices, avoiding expensive data loading and statistical computations.

    Args:
        volume_path: Path to NIFTI volume file

    Returns:
        Dictionary containing essential volume metadata

    Raises:
        ValueError: If volume cannot be loaded or is invalid
    """
    try:
        # Load NIFTI volume (header only, no pixel data loading)
        nifti_img = nib.load(volume_path)
        header = nifti_img.header

        # Essential metadata for 3D volume reconstruction
        metadata = {
            "file_path": str(volume_path),
            "shape": tuple(nifti_img.shape),
            "voxel_spacing": tuple(header.get_zooms()),
            "orientation": str(nib.aff2axcodes(nifti_img.affine)),
            "affine_matrix": nifti_img.affine.tolist(),
            "dtype": str(nifti_img.get_data_dtype()),
        }

        logger.debug(f"Extracted metadata for: {volume_path.name}")
        return metadata

    except Exception as e:
        raise ValueError(f"Failed to extract metadata from {volume_path}: {e}")


def validate_volume_consistency(
    volume_metadata: list[dict], tolerance: float = 0.1
) -> dict[str, bool]:
    """Validate consistency across multiple volumes.

    Args:
        volume_metadata: List of metadata dictionaries
        tolerance: Tolerance for spacing/size comparisons (fraction)

    Returns:
        Dictionary of consistency checks
    """
    if not volume_metadata:
        return {"consistent": False, "reason": "No volumes provided"}

    if len(volume_metadata) == 1:
        return {"consistent": True, "single_volume": True}

    checks = {
        "consistent_shape": True,
        "consistent_spacing": True,
        "consistent_orientation": True,
        "consistent_dtype": True,
        "reason": [],
    }

    # Reference volume (first one)
    ref = volume_metadata[0]
    ref_shape = ref["shape"]
    ref_spacing = ref["voxel_spacing"]
    ref_orientation = ref["orientation"]
    ref_dtype = ref["dtype"]

    for i, meta in enumerate(volume_metadata[1:], 1):
        # Check shape consistency
        if meta["shape"] != ref_shape:
            checks["consistent_shape"] = False
            checks["reason"].append(
                f"Volume {i} shape {meta['shape']} != reference {ref_shape}"
            )

        # Check spacing consistency (with tolerance)
        spacing_diff = np.array(meta["voxel_spacing"]) - np.array(ref_spacing)
        relative_diff = np.abs(spacing_diff) / np.array(ref_spacing)
        if np.any(relative_diff > tolerance):
            checks["consistent_spacing"] = False
            checks["reason"].append(
                f"Volume {i} spacing {meta['voxel_spacing']} differs from reference {ref_spacing}"
            )

        # Check orientation consistency
        if meta["orientation"] != ref_orientation:
            checks["consistent_orientation"] = False
            checks["reason"].append(
                f"Volume {i} orientation {meta['orientation']} != reference {ref_orientation}"
            )

        # Check dtype consistency
        if meta["dtype"] != ref_dtype:
            checks["consistent_dtype"] = False
            checks["reason"].append(
                f"Volume {i} dtype {meta['dtype']} != reference {ref_dtype}"
            )

    # Overall consistency
    checks["consistent"] = all(
        [
            checks["consistent_shape"],
            checks["consistent_spacing"],
            checks["consistent_orientation"],
            checks["consistent_dtype"],
        ]
    )

    return checks


def log_metadata_summary(metadata: dict) -> None:
    """Log a summary of volume metadata.

    Args:
        metadata: Volume metadata dictionary
    """
    logger.info(f"Volume: {Path(metadata['file_path']).name}")
    logger.info(f"  Shape: {metadata['shape']}")
    logger.info(f"  Spacing: {metadata['voxel_spacing']}")
    logger.info(f"  Orientation: {metadata['orientation']}")
    logger.info(f"  Data type: {metadata['dtype']}")
