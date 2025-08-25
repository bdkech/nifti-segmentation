"""Inference and prediction module for volume reconstruction."""

from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import torch
from loguru import logger
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Invertd

from ..data.transforms import create_inference_pipeline
from ..utils.config import Config
from .unetr import UNet2DModel


class VolumePredictor:
    """High-level interface for volume segmentation prediction.
    
    This class handles:
    - Model loading from checkpoints
    - Volume preprocessing for inference
    - Sliding window inference for large volumes
    - Post-processing and volume reconstruction
    - Output saving in NIFTI format
    """
    
    def __init__(
        self,
        model_path: Path,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize volume predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Optional configuration (will load from checkpoint if None)
            device: Device for inference (auto-detect if None)
            
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Setup device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(config)
        self.model.eval()
        
        # Setup preprocessing pipeline
        self.preprocess_transforms = self._setup_preprocessing()
        self.inverse_transforms = None  # Will be set during preprocessing
        
        logger.info("Volume predictor initialized successfully")
    
    def _load_model(self, config: Optional[Config] = None) -> UNet2DModel:
        """Load model from checkpoint.
        
        Args:
            config: Optional configuration
            
        Returns:
            Loaded UNetR model
        """
        try:
            if config:
                model = UNet2DModel.load_from_checkpoint(
                    str(self.model_path),
                    model_config=config.model,
                    training_config=config.training,
                    map_location=self.device,
                )
            else:
                # Load with automatic config detection
                model = UNet2DModel.load_from_checkpoint(
                    str(self.model_path),
                    map_location=self.device,
                )
            
            model.to(self.device)
            logger.info(f"Model loaded from: {self.model_path}")
            return model
            
        except Exception as e:
            logger.exception(f"Failed to load model from {self.model_path}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _setup_preprocessing(self) -> Compose:
        """Setup preprocessing pipeline for inference.
        
        Returns:
            Preprocessing transform pipeline
        """
        # Extract target parameters from model config
        if hasattr(self.model, 'model_config'):
            target_size = self.model.model_config.img_size
            target_spacing = getattr(self.model.model_config, 'target_spacing', None)
        else:
            target_size = [224, 224, 32]  # Default size
            target_spacing = None
        
        return create_inference_pipeline(
            target_spacing=target_spacing,
            target_size=target_size,
        )
    
    def predict_volume(
        self,
        volume_path: Path,
        output_path: Optional[Path] = None,
        roi_size: Optional[list[int]] = None,
        overlap: float = 0.5,
        sw_batch_size: int = 4,
        save_probability: bool = False,
        apply_postprocessing: bool = True,
    ) -> dict[str, Any]:
        """Predict segmentation for a single volume.
        
        Args:
            volume_path: Path to input NIFTI volume
            output_path: Optional path to save prediction (auto-generated if None)
            roi_size: Region of interest size for sliding window
            overlap: Overlap between sliding windows (0-1)
            sw_batch_size: Batch size for sliding windows
            save_probability: Whether to save probability maps
            apply_postprocessing: Whether to apply post-processing
            
        Returns:
            Dictionary containing prediction results and metadata
            
        Raises:
            FileNotFoundError: If input volume doesn't exist
            RuntimeError: If prediction fails
        """
        volume_path = Path(volume_path)
        if not volume_path.exists():
            raise FileNotFoundError(f"Input volume not found: {volume_path}")
        
        logger.info(f"Predicting segmentation for: {volume_path}")
        
        try:
            # 1. Preprocess volume
            logger.debug("Preprocessing volume...")
            data = self._preprocess_volume(volume_path)
            
            # 2. Run inference
            logger.debug("Running inference...")
            prediction = self._run_inference(
                data,
                roi_size=roi_size,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
            )
            
            # 3. Post-process prediction
            if apply_postprocessing:
                logger.debug("Post-processing prediction...")
                prediction = self._postprocess_prediction(data, prediction)
            
            # 4. Save results
            results = {}
            if output_path or save_probability:
                results = self._save_prediction(
                    data,
                    prediction,
                    volume_path,
                    output_path,
                    save_probability,
                )
            
            # 5. Compute prediction statistics
            pred_stats = self._compute_prediction_stats(prediction)
            results.update(pred_stats)
            
            logger.info("Prediction completed successfully")
            return results
            
        except Exception as e:
            logger.exception(f"Prediction failed for {volume_path}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _preprocess_volume(self, volume_path: Path) -> dict[str, Any]:
        """Preprocess volume for inference.
        
        Args:
            volume_path: Path to input volume
            
        Returns:
            Preprocessed data dictionary
        """
        data = {"image": str(volume_path)}
        processed_data = self.preprocess_transforms(data)
        
        # Setup inverse transforms for post-processing
        self.inverse_transforms = Invertd(
            keys=["pred"],
            transform=self.preprocess_transforms,
            orig_keys="image",
            meta_keys=None,
            orig_meta_keys=None,
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        )
        
        return processed_data
    
    def _run_inference(
        self,
        data: dict[str, Any],
        roi_size: Optional[list[int]] = None,
        overlap: float = 0.5,
        sw_batch_size: int = 4,
    ) -> torch.Tensor:
        """Run sliding window inference on preprocessed volume.
        
        Args:
            data: Preprocessed data dictionary
            roi_size: Region of interest size for sliding window
            overlap: Overlap between sliding windows
            sw_batch_size: Batch size for sliding windows
            
        Returns:
            Prediction tensor
        """
        image = data["image"].unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Use model's default roi_size if not specified
        if roi_size is None:
            if hasattr(self.model, 'model_config'):
                roi_size = self.model.model_config.img_size
            else:
                roi_size = [224, 224, 32]
        
        # Run sliding window inference
        with torch.no_grad():
            prediction = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=overlap,
                mode="gaussian",  # Use Gaussian weighting for overlaps
                sigma_scale=0.125,
                padding_mode="constant",
                cval=0.0,
                sw_device=self.device,
                device=self.device,
            )
        
        return prediction.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    
    def _postprocess_prediction(
        self,
        data: dict[str, Any],
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Apply post-processing to prediction.
        
        Args:
            data: Original preprocessed data
            prediction: Raw model prediction
            
        Returns:
            Post-processed prediction
        """
        # Apply model's post-processing transforms
        if hasattr(self.model, 'post_pred'):
            prediction = self.model.post_pred(prediction.unsqueeze(0)).squeeze(0)
        
        # Apply inverse transforms to restore original space
        if self.inverse_transforms:
            # Prepare data for inverse transform
            post_data = dict(data)
            post_data["pred"] = prediction
            
            try:
                # Apply inverse transforms
                post_data = self.inverse_transforms(post_data)
                prediction = post_data["pred"]
            except Exception as e:
                logger.warning(f"Inverse transform failed: {e}, keeping prediction as-is")
        
        return prediction
    
    def _save_prediction(
        self,
        data: dict[str, Any],
        prediction: torch.Tensor,
        input_path: Path,
        output_path: Optional[Path] = None,
        save_probability: bool = False,
    ) -> dict[str, Any]:
        """Save prediction to NIFTI file.
        
        Args:
            data: Preprocessed data dictionary
            prediction: Final prediction tensor
            input_path: Path to input volume
            output_path: Optional output path
            save_probability: Whether to save probability maps
            
        Returns:
            Dictionary with save paths and metadata
        """
        results = {}
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.with_name(
                f"{input_path.stem}_segmentation.nii.gz"
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        if isinstance(prediction, torch.Tensor):
            pred_numpy = prediction.numpy()
        else:
            pred_numpy = np.array(prediction)
        
        # Remove channel dimension if present
        if pred_numpy.ndim == 4 and pred_numpy.shape[0] == 1:
            pred_numpy = pred_numpy.squeeze(0)
        
        # Get original affine if available
        if "image_meta_dict" in data and "affine" in data["image_meta_dict"]:
            affine = data["image_meta_dict"]["affine"]
        else:
            affine = np.eye(4)  # Identity affine as fallback
        
        # Save segmentation (binary)
        if not save_probability:
            # Convert to binary segmentation
            binary_pred = (pred_numpy > 0.5).astype(np.uint8)
            nii_img = nib.Nifti1Image(binary_pred, affine)
            nib.save(nii_img, str(output_path))
            results["segmentation_path"] = str(output_path)
            logger.info(f"Segmentation saved to: {output_path}")
        
        # Save probability maps
        if save_probability:
            prob_path = output_path.with_name(
                f"{output_path.stem}_probability.nii.gz"
            )
            # Ensure probabilities are in [0, 1] range
            prob_pred = np.clip(pred_numpy, 0, 1).astype(np.float32)
            nii_img = nib.Nifti1Image(prob_pred, affine)
            nib.save(nii_img, str(prob_path))
            results["probability_path"] = str(prob_path)
            logger.info(f"Probability map saved to: {prob_path}")
        
        return results
    
    def _compute_prediction_stats(
        self,
        prediction: torch.Tensor,
    ) -> dict[str, Any]:
        """Compute statistics for the prediction.
        
        Args:
            prediction: Prediction tensor
            
        Returns:
            Dictionary of prediction statistics
        """
        if isinstance(prediction, torch.Tensor):
            pred_numpy = prediction.numpy()
        else:
            pred_numpy = np.array(prediction)
        
        # Remove channel dimension if present
        if pred_numpy.ndim == 4 and pred_numpy.shape[0] == 1:
            pred_numpy = pred_numpy.squeeze(0)
        
        # Compute statistics
        stats = {
            "prediction_shape": pred_numpy.shape,
            "min_value": float(pred_numpy.min()),
            "max_value": float(pred_numpy.max()),
            "mean_value": float(pred_numpy.mean()),
            "std_value": float(pred_numpy.std()),
        }
        
        # Binary segmentation statistics
        binary_pred = pred_numpy > 0.5
        stats.update({
            "positive_voxels": int(binary_pred.sum()),
            "total_voxels": int(binary_pred.size),
            "positive_fraction": float(binary_pred.mean()),
        })
        
        # Connected component analysis
        try:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(binary_pred)
            stats["num_connected_components"] = int(num_features)
            
            if num_features > 0:
                # Size of largest component
                component_sizes = [
                    (labeled_array == i).sum() for i in range(1, num_features + 1)
                ]
                stats["largest_component_size"] = int(max(component_sizes))
                stats["mean_component_size"] = float(np.mean(component_sizes))
        except ImportError:
            logger.debug("SciPy not available for connected component analysis")
        
        return {"prediction_stats": stats}
    
    def predict_batch(
        self,
        volume_paths: list[Path],
        output_dir: Path,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Predict segmentation for multiple volumes.
        
        Args:
            volume_paths: List of input volume paths
            output_dir: Directory to save predictions
            **kwargs: Additional arguments for predict_volume
            
        Returns:
            List of prediction results for each volume
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, volume_path in enumerate(volume_paths):
            logger.info(f"Processing volume {i+1}/{len(volume_paths)}: {volume_path}")
            
            try:
                # Generate output path
                output_path = output_dir / f"{volume_path.stem}_segmentation.nii.gz"
                
                # Predict single volume
                result = self.predict_volume(
                    volume_path=volume_path,
                    output_path=output_path,
                    **kwargs,
                )
                
                result["input_path"] = str(volume_path)
                result["success"] = True
                results.append(result)
                
            except Exception as e:
                logger.exception(f"Failed to process {volume_path}")
                results.append({
                    "input_path": str(volume_path),
                    "success": False,
                    "error": str(e),
                })
        
        # Log summary
        successful = sum(1 for r in results if r.get("success", False))
        logger.info(f"Batch prediction completed: {successful}/{len(volume_paths)} successful")
        
        return results


def create_predictor(
    model_path: Path,
    config: Optional[Config] = None,
    device: Optional[str] = None,
) -> VolumePredictor:
    """Create volume predictor instance.
    
    Args:
        model_path: Path to trained model checkpoint
        config: Optional configuration
        device: Optional device specification
        
    Returns:
        Initialized VolumePredictor instance
    """
    device_obj = None
    if device:
        device_obj = torch.device(device)
    
    return VolumePredictor(
        model_path=model_path,
        config=config,
        device=device_obj,
    )