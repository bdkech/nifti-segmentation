# NiftiLearn - Medical Image Segmentation Pipeline

A production-ready pipeline for medical image segmentation using slice-based 2D UNet architecture with PyTorch Lightning, designed for high-performance workflows with Docker-like CLI experience.

## 🎯 Key Features

- ✅ **Slice-Based Processing**: Efficient 2D UNet on volume slices (one volume per batch)
- ✅ **MONAI Integration**: Built-in loss functions and metrics optimized for medical imaging
- ✅ **PyTorch Lightning**: Professional training with multi-GPU support, checkpointing, early stopping
- ✅ **Hounsfield Unit Processing**: Adaptive and preset-based HU normalization
- ✅ **Volume Reconstruction**: Automatic 2D-to-3D reassembly with metadata preservation
- ✅ **CLI Interface**: Docker-style commands with comprehensive configuration support
- ✅ **Production Ready**: Complete error handling, logging, and type safety

## 📁 Project Structure

```
niftilearn/
├── cli/                        # Command Line Interface
│   ├── __init__.py
│   └── main.py                 # Click-based CLI with train/predict commands
├── config/                     # Configuration Management
│   ├── __init__.py
│   ├── loader.py               # YAML configuration loading
│   └── models.py               # Pydantic configuration models
├── core/                       # Core Utilities
│   ├── __init__.py
│   ├── logging.py              # Loguru logging setup
│   └── metadata.py             # NIFTI metadata extraction
├── data/                       # Data Pipeline
│   ├── __init__.py
│   ├── datamodule.py           # Lightning DataModule
│   ├── datasets.py             # Volume-based PyTorch datasets
│   ├── loaders.py              # NIFTI discovery and loading
│   └── transforms.py           # Slice extraction and HU processing
├── models/                     # Model Architecture
│   ├── __init__.py
│   └── unet.py                 # UNet2D Lightning module with training
├── training/                   # Training Components
│   ├── __init__.py
│   ├── losses.py               # MONAI loss functions (Dice, Focal, DiceCE)
│   └── metrics.py              # MONAI segmentation metrics
└── utils/                      # Utilities
    ├── __init__.py
    └── reconstruction.py       # 2D-to-3D volume reconstruction
```

## 🚀 Usage

### CLI Commands
```bash
# Train model with configuration file
niftilearn --config example_config.yaml train

# Train with parameter overrides
niftilearn --config example_config.yaml train --epochs 50 --batch-size 4 --learning-rate 5e-5

# Multi-GPU training with torchrun
torchrun --nproc_per_node=2 -m niftilearn.cli.main --config example_config.yaml train

# Generate predictions on new volumes
niftilearn predict --model checkpoints/best.ckpt --input volume.nii.gz --output segmentation.nii.gz

# Prediction with preprocessing config
niftilearn --config example_config.yaml predict --model checkpoints/best.ckpt --input volume.nii.gz --output segmentation.nii.gz
```

### Python API
```python
# Core imports
from niftilearn.config.loader import load_config
from niftilearn.data.datamodule import NiftiDataModule  
from niftilearn.models.unet import UNet2D
from niftilearn.utils.reconstruction import reconstruct_volume_from_predictions

# Load configuration and create components
config = load_config("example_config.yaml")
model = UNet2D(config.model)
datamodule = NiftiDataModule(config.data)

# Training with Lightning
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=config.training.epochs)
trainer.fit(model, datamodule)
```

## 🏥 Medical Image Processing

The pipeline includes specialized processing for medical NIFTI volumes:

### Hounsfield Unit Normalization
- **Adaptive HU Normalization**: Automatic percentile-based windowing for robust preprocessing
- **Fixed HU Windows**: Support for clinical presets (soft tissue, bone, lung)
- **Volume-to-Slice Processing**: Extract 2D slices from 3D volumes along configurable axes

### Slice-Based Architecture
- **One Volume Per Batch**: Process entire volumes as batches of 2D slices
- **Configurable Slice Axis**: Extract slices along axial (0), coronal (1), or sagittal (2) planes
- **Automatic Reconstruction**: Reassemble 2D predictions into 3D volumes with preserved metadata

```yaml
# Configuration example
data:
  slice_axis: 2                        # Sagittal slices
  use_adaptive_hu_normalization: true  # Adaptive HU windowing
  adaptive_hu_lower_percentile: 0.5    # Lower percentile for windowing
  adaptive_hu_upper_percentile: 99.5   # Upper percentile for windowing
  target_spacing: [1.0, 1.0, 1.0]     # Resample to 1mm isotropic
  img_size: [224, 224]                 # 2D slice size
```

## 🔧 Installation

```bash
# Install dependencies with uv (recommended)
uv add monai torch lightning click loguru nibabel wandb numpy scipy pydantic pyyaml rich

# Install in development mode
uv pip install -e .

# Or with pip
pip install -e .
```

## 📝 Configuration

Use YAML configuration files for reproducible experiments. See `example_config.yaml` for complete example:

```yaml
data:
  data_dir: "./data/volumes"
  annotation_dir: "./data/annotations"
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
  slice_axis: 2                        # Extract sagittal slices
  batch_size: 8                        # Slices per batch from same volume
  img_size: [224, 224]                 # Target 2D slice size
  use_adaptive_hu_normalization: true  # Adaptive HU windowing

model:
  img_size: [224, 224]                 # Must match data.img_size
  in_channels: 1
  out_channels: 1  
  features: [32, 32, 64, 128, 256, 32] # UNet feature channels
  activation: "PRELU"

training:
  epochs: 100
  batch_size: 8                        # Must match data.batch_size
  learning_rate: 1e-4
  optimizer: "AdamW"
  loss_function: "dicece"              # MONAI loss functions
  patience: 15                         # Early stopping patience

compute:
  devices: "auto"                      # GPU devices to use
  accelerator: "auto"                  # gpu/cpu/mps
  precision: "16-mixed"                # Mixed precision training

wandb:
  enabled: false                       # Optional W&B logging
  project: "nifti-segmentation"

output_dir: "./outputs"
```

## 📊 Training & Metrics

The pipeline automatically tracks comprehensive metrics during training:

- **Loss Functions**: Dice Loss, Focal Loss, Combined Dice+CrossEntropy (MONAI implementations)
- **Metrics**: Dice coefficient, IoU (Intersection over Union), pixel accuracy
- **Checkpointing**: Automatic saving of best models based on validation loss
- **Early Stopping**: Configurable patience to prevent overfitting
- **Multi-GPU**: Automatic distributed training with Lightning + torchrun

## 🔍 Data Structure

Expected directory structure for training:

```text
data/
├── volumes/
│   ├── subject_001/
│   │   └── study_volume/
│   │       └── volume.nii.gz
│   └── subject_002/
│       └── study_volume/
│           └── volume.nii.gz
└── annotations/
    ├── Segmentations.subject_001.ART.nii.gz
    ├── Segmentations.subject_001.RA.nii.gz
    ├── Segmentations.subject_002.ART.nii.gz
    └── Segmentations.subject_002.RA.nii.gz
```

Annotation types: `ART` (artery), `RA` (rectus abdominis), `S_FAT` (subcutaneous fat)
