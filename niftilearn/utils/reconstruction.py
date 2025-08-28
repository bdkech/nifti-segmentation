"""Volume reconstruction utilities for reassembling 2D slice predictions into 3D volumes.

This module provides functions to reconstruct 3D volumes from 2D slice predictions,
handling proper spacing, orientation, and NIFTI format compatibility.
"""

from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
from loguru import logger


class VolumeReconstructor:
    """Reconstructs 3D volumes from 2D slice predictions.
    
    This class handles the reassembly of 2D slice predictions back into 3D volumes
    while preserving the original NIFTI metadata, spacing, and orientation.
    """

    def __init__(
        self,
        slice_axis: int = 2,
        preserve_metadata: bool = True,
    ) -> None:
        """Initialize volume reconstructor.
        
        Args:
            slice_axis: Axis along which slices were extracted (0=axial, 1=coronal, 2=sagittal)
            preserve_metadata: Whether to preserve original NIFTI metadata
        """
        self.slice_axis = slice_axis
        self.preserve_metadata = preserve_metadata
        
        logger.debug(f"VolumeReconstructor initialized with slice_axis={slice_axis}")

    def reconstruct_volume(
        self,
        slice_predictions: torch.Tensor,
        slice_indices: list[int],
        reference_volume_path: Union[str, Path],
        target_shape: Optional[tuple[int, int, int]] = None,
    ) -> tuple[np.ndarray, nib.Nifti1Image]:
        """Reconstruct 3D volume from 2D slice predictions.
        
        Args:
            slice_predictions: Tensor of slice predictions [N, C, H, W]
            slice_indices: Original slice indices in the volume
            reference_volume_path: Path to reference volume for metadata
            target_shape: Target 3D shape (if None, inferred from reference)
            
        Returns:
            Tuple of (reconstructed_array, nifti_image)
            
        Raises:
            ValueError: If slice predictions and indices don't match
            FileNotFoundError: If reference volume doesn't exist
        """
        # Validate inputs
        if len(slice_predictions) != len(slice_indices):
            raise ValueError(
                f"Number of predictions ({len(slice_predictions)}) doesn't match "
                f"number of indices ({len(slice_indices)})"
            )
            
        if not Path(reference_volume_path).exists():
            raise FileNotFoundError(f"Reference volume not found: {reference_volume_path}")
        
        # Load reference volume for metadata
        reference_img = nib.load(reference_volume_path)
        reference_data = reference_img.get_fdata()
        
        # Determine target shape
        if target_shape is None:
            target_shape = reference_data.shape
        
        logger.info(f"Reconstructing volume with shape {target_shape} from {len(slice_predictions)} slices")
        
        # Convert predictions to numpy
        predictions_np = slice_predictions.cpu().numpy()
        
        # Handle channel dimension (assume single channel output)
        if predictions_np.ndim == 4 and predictions_np.shape[1] == 1:
            predictions_np = predictions_np.squeeze(1)  # Remove channel dimension
        elif predictions_np.ndim == 4:
            logger.warning(f"Multi-channel predictions ({predictions_np.shape[1]} channels), using first channel")
            predictions_np = predictions_np[:, 0, :, :]
        
        # Initialize output volume
        reconstructed_volume = np.zeros(target_shape, dtype=predictions_np.dtype)
        
        # Place slices in correct positions
        for i, slice_idx in enumerate(slice_indices):
            if slice_idx >= target_shape[self.slice_axis]:
                logger.warning(f"Slice index {slice_idx} exceeds volume size {target_shape[self.slice_axis]}")
                continue
                
            # Get prediction slice
            pred_slice = predictions_np[i]
            
            # Resize if necessary to match target shape
            target_slice_shape = self._get_slice_shape(target_shape, self.slice_axis)
            if pred_slice.shape != target_slice_shape:
                pred_slice = self._resize_slice(pred_slice, target_slice_shape)
            
            # Place slice in volume
            if self.slice_axis == 0:
                reconstructed_volume[slice_idx, :, :] = pred_slice
            elif self.slice_axis == 1:
                reconstructed_volume[:, slice_idx, :] = pred_slice
            elif self.slice_axis == 2:
                reconstructed_volume[:, :, slice_idx] = pred_slice
        
        # Create NIFTI image with preserved metadata
        if self.preserve_metadata:
            reconstructed_img = nib.Nifti1Image(
                reconstructed_volume,
                reference_img.affine,
                reference_img.header.copy()
            )
            # Update header for segmentation mask
            reconstructed_img.header.set_data_dtype(reconstructed_volume.dtype)
        else:
            # Create basic NIFTI image
            reconstructed_img = nib.Nifti1Image(reconstructed_volume, np.eye(4))
        
        logger.info(f"Volume reconstruction completed: shape={reconstructed_volume.shape}")
        return reconstructed_volume, reconstructed_img

    def _get_slice_shape(self, volume_shape: tuple[int, int, int], axis: int) -> tuple[int, int]:
        """Get the 2D shape of a slice from a 3D volume shape.
        
        Args:
            volume_shape: 3D volume shape (H, W, D)
            axis: Slice axis
            
        Returns:
            2D slice shape
        """
        if axis == 0:
            return (volume_shape[1], volume_shape[2])  # (W, D)
        elif axis == 1:
            return (volume_shape[0], volume_shape[2])  # (H, D)
        elif axis == 2:
            return (volume_shape[0], volume_shape[1])  # (H, W)
        else:
            raise ValueError(f"Invalid slice axis: {axis}")

    def _resize_slice(self, slice_data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Resize a 2D slice to target shape using interpolation.
        
        Args:
            slice_data: 2D slice data
            target_shape: Target 2D shape
            
        Returns:
            Resized slice data
        """
        # Use scipy for resizing if available, otherwise use simple nearest neighbor
        try:
            from scipy.ndimage import zoom
            
            # Calculate zoom factors
            zoom_factors = (
                target_shape[0] / slice_data.shape[0],
                target_shape[1] / slice_data.shape[1],
            )
            
            # Use nearest neighbor for segmentation masks
            resized_slice = zoom(slice_data, zoom_factors, order=0)
            
            # Ensure exact target shape (zoom might have small rounding errors)
            if resized_slice.shape != target_shape:
                # Crop or pad as needed
                resized_slice = self._crop_or_pad(resized_slice, target_shape)
                
            return resized_slice
            
        except ImportError:
            logger.warning("SciPy not available, using simple resizing")
            # Simple nearest neighbor resizing
            return self._simple_resize(slice_data, target_shape)

    def _crop_or_pad(self, data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Crop or pad data to match target shape.
        
        Args:
            data: Input 2D array
            target_shape: Target shape
            
        Returns:
            Cropped or padded array
        """
        current_shape = data.shape
        result = np.zeros(target_shape, dtype=data.dtype)
        
        # Calculate crop/pad regions
        h_min = min(current_shape[0], target_shape[0])
        w_min = min(current_shape[1], target_shape[1])
        
        # Copy data
        result[:h_min, :w_min] = data[:h_min, :w_min]
        
        return result

    def _simple_resize(self, data: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Simple nearest neighbor resizing without external dependencies.
        
        Args:
            data: Input 2D array
            target_shape: Target shape
            
        Returns:
            Resized array
        """
        h_ratio = data.shape[0] / target_shape[0]
        w_ratio = data.shape[1] / target_shape[1]
        
        result = np.zeros(target_shape, dtype=data.dtype)
        
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                src_i = int(i * h_ratio)
                src_j = int(j * w_ratio)
                src_i = min(src_i, data.shape[0] - 1)
                src_j = min(src_j, data.shape[1] - 1)
                result[i, j] = data[src_i, src_j]
        
        return result

    def save_volume(
        self,
        nifti_image: nib.Nifti1Image,
        output_path: Union[str, Path],
    ) -> None:
        """Save reconstructed volume to disk.
        
        Args:
            nifti_image: NIFTI image with metadata
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save NIFTI image
        nib.save(nifti_image, str(output_path))
        
        logger.info(f"Reconstructed volume saved: {output_path}")


def reconstruct_volume_from_predictions(
    predictions: torch.Tensor,
    slice_indices: list[int],
    reference_volume: Union[str, Path],
    output_path: Union[str, Path],
    slice_axis: int = 2,
) -> Path:
    """Convenience function to reconstruct and save a volume from predictions.
    
    Args:
        predictions: Tensor of slice predictions [N, C, H, W]
        slice_indices: Original slice indices
        reference_volume: Path to reference volume for metadata
        output_path: Output file path
        slice_axis: Slice extraction axis
        
    Returns:
        Path to saved volume
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If reference volume doesn't exist
    """
    reconstructor = VolumeReconstructor(slice_axis=slice_axis)
    
    # Reconstruct volume
    volume_data, nifti_image = reconstructor.reconstruct_volume(
        slice_predictions=predictions,
        slice_indices=slice_indices,
        reference_volume_path=reference_volume,
    )
    
    # Save volume
    output_path = Path(output_path)
    reconstructor.save_volume(nifti_image, output_path)
    
    return output_path