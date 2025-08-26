"""Volume metadata utilities for NiftiLearn."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from loguru import logger


def extract_volume_metadata(volume_path: Path) -> Dict[str, Union[str, int, float, List, Tuple]]:
    """Extract metadata from a NIFTI volume.
    
    Args:
        volume_path: Path to NIFTI volume file
        
    Returns:
        Dictionary containing volume metadata
        
    Raises:
        ValueError: If volume cannot be loaded or is invalid
    """
    try:
        # Load NIFTI volume
        nifti_img = nib.load(volume_path)
        header = nifti_img.header
        data = nifti_img.get_fdata()
        
        # Basic volume properties
        metadata = {
            "file_path": str(volume_path),
            "file_size_mb": volume_path.stat().st_size / (1024 * 1024),
            "shape": tuple(data.shape),
            "ndim": data.ndim,
            "dtype": str(data.dtype),
            "data_min": float(np.min(data)),
            "data_max": float(np.max(data)),
            "data_mean": float(np.mean(data)),
            "data_std": float(np.std(data)),
        }
        
        # Voxel spacing and orientation
        metadata.update({
            "voxel_spacing": tuple(header.get_zooms()),
            "orientation": str(nib.aff2axcodes(nifti_img.affine)),
            "affine_matrix": nifti_img.affine.tolist(),
        })
        
        # Header-specific information
        metadata.update({
            "units": {
                "spatial": header.get_xyzt_units()[0],
                "temporal": header.get_xyzt_units()[1],
            },
            "description": str(header.get("descrip", b"").decode("utf-8", errors="ignore")),
        })
        
        # Hounsfield Unit statistics (for CT data)
        if is_likely_ct_data(data):
            hu_stats = calculate_hu_statistics(data)
            metadata["hounsfield_units"] = hu_stats
        
        logger.debug(f"Extracted metadata for: {volume_path.name}")
        return metadata
        
    except Exception as e:
        raise ValueError(f"Failed to extract metadata from {volume_path}: {e}")


def calculate_hu_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate Hounsfield Unit statistics for CT data.
    
    Args:
        data: Volume data array
        
    Returns:
        Dictionary with HU statistics
    """
    # Remove background/air voxels (typical HU threshold)
    tissue_mask = data > -1000
    tissue_data = data[tissue_mask]
    
    if len(tissue_data) == 0:
        logger.warning("No tissue voxels found for HU statistics")
        tissue_data = data[data > data.min()]
    
    hu_stats = {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "tissue_min": float(np.min(tissue_data)) if len(tissue_data) > 0 else float(np.min(data)),
        "tissue_max": float(np.max(tissue_data)) if len(tissue_data) > 0 else float(np.max(data)),
        "tissue_mean": float(np.mean(tissue_data)) if len(tissue_data) > 0 else float(np.mean(data)),
        "percentile_1": float(np.percentile(data, 1)),
        "percentile_5": float(np.percentile(data, 5)),
        "percentile_95": float(np.percentile(data, 95)),
        "percentile_99": float(np.percentile(data, 99)),
    }
    
    return hu_stats


def is_likely_ct_data(data: np.ndarray) -> bool:
    """Determine if volume data is likely from a CT scan.
    
    Args:
        data: Volume data array
        
    Returns:
        True if data appears to be CT (Hounsfield Units)
    """
    # CT data typically has negative values (air) and a wide range
    has_negative = np.any(data < -100)
    data_range = np.max(data) - np.min(data)
    wide_range = data_range > 1000
    
    # Typical HU range indicators
    air_present = np.any(data < -800)  # Air around -1000 HU
    bone_present = np.any(data > 200)  # Bone typically > 200 HU
    
    return has_negative and wide_range and (air_present or bone_present)


def validate_volume_consistency(
    volume_metadata: List[Dict], 
    tolerance: float = 0.1
) -> Dict[str, bool]:
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
            checks["reason"].append(f"Volume {i} shape {meta['shape']} != reference {ref_shape}")
        
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
    checks["consistent"] = all([
        checks["consistent_shape"],
        checks["consistent_spacing"], 
        checks["consistent_orientation"],
        checks["consistent_dtype"],
    ])
    
    return checks


def find_optimal_hu_window(
    volume_metadata_list: List[Dict], 
    lower_percentile: float = 0.5, 
    upper_percentile: float = 99.5
) -> Optional[Tuple[float, float]]:
    """Find optimal HU window across multiple volumes.
    
    Args:
        volume_metadata_list: List of volume metadata containing HU statistics
        lower_percentile: Lower percentile for windowing
        upper_percentile: Upper percentile for windowing
        
    Returns:
        Tuple of (hu_min, hu_max) or None if no HU data available
    """
    hu_data = []
    
    for meta in volume_metadata_list:
        if "hounsfield_units" in meta:
            hu_stats = meta["hounsfield_units"]
            # Collect percentile values from each volume
            hu_data.append(hu_stats[f"percentile_{int(lower_percentile)}"])
            hu_data.append(hu_stats[f"percentile_{int(upper_percentile)}"])
    
    if not hu_data:
        logger.warning("No Hounsfield Unit data found in metadata")
        return None
    
    # Use overall percentiles across all data
    hu_min = float(np.percentile(hu_data, lower_percentile))
    hu_max = float(np.percentile(hu_data, upper_percentile))
    
    logger.info(f"Optimal HU window: [{hu_min:.1f}, {hu_max:.1f}]")
    return (hu_min, hu_max)


def log_metadata_summary(metadata: Dict) -> None:
    """Log a summary of volume metadata.
    
    Args:
        metadata: Volume metadata dictionary
    """
    logger.info(f"Volume: {Path(metadata['file_path']).name}")
    logger.info(f"  Shape: {metadata['shape']}")
    logger.info(f"  Spacing: {metadata['voxel_spacing']}")
    logger.info(f"  Orientation: {metadata['orientation']}")
    logger.info(f"  Data range: [{metadata['data_min']:.2f}, {metadata['data_max']:.2f}]")
    logger.info(f"  File size: {metadata['file_size_mb']:.1f} MB")
    
    if "hounsfield_units" in metadata:
        hu = metadata["hounsfield_units"]
        logger.info(f"  HU range: [{hu['min']:.1f}, {hu['max']:.1f}]")
        logger.info(f"  HU tissue: [{hu['tissue_min']:.1f}, {hu['tissue_max']:.1f}]")