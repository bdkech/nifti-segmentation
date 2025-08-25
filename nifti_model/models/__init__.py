"""Model architecture and inference submodule.

This submodule provides:
- 2D UNet model implementation with Lightning integration
- Slice-based prediction and volume reconstruction capabilities  
- Individual slice inference for efficient processing
"""

from .inference import VolumePredictor, create_predictor
from .unetr import UNet2DModel

__all__ = [
    "UNet2DModel",
    "VolumePredictor", 
    "create_predictor",
]