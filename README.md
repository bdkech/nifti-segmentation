# NIFTI Model - Medical Volume Segmentation Pipeline

A comprehensive pipeline for 3D medical volume segmentation using UNetR architecture with Hounsfield Unit support.

## 📁 Project Structure

```
nifti_model/
├── __init__.py                 # Main package with core imports
├── cli/                        # Command Line Interface
│   ├── __init__.py
│   └── main.py                 # Click-based CLI commands
├── data/                       # Data Loading & Preprocessing
│   ├── __init__.py
│   ├── loaders.py              # NIFTI dataset and data modules
│   └── transforms.py           # HU normalization & preprocessing
├── models/                     # Model Architecture & Inference
│   ├── __init__.py
│   ├── unetr.py                # UNetR Lightning module
│   └── inference.py            # Volume prediction pipeline
├── training/                   # Training & Experiment Tracking
│   ├── __init__.py
│   ├── trainer.py              # Lightning training manager
│   └── logging.py              # Weights & Biases integration
└── utils/                      # Configuration & Utilities
    ├── __init__.py
    ├── config.py               # Pydantic configuration classes
    └── logging.py              # Loguru logging setup
```

## 🚀 Usage

### Import Core Components
```python
from nifti_model import (
    NiftiVolumeDataset, NiftiDataModule,
    UNetRModel, VolumePredictor, 
    TrainingManager, Config
)
```

### Import Specific Submodules
```python
# Data processing
from nifti_model.data import (
    HounsfieldUnitNormalizationd,
    create_preprocessing_pipeline
)

# Model components  
from nifti_model.models import UNetRModel, create_predictor

# Training utilities
from nifti_model.training import TrainingManager, setup_wandb

# Configuration
from nifti_model.utils import Config, DataConfig, ModelConfig
```

### CLI Commands
```bash
# Preprocess NIFTI volumes
nifti-model preprocess -d ./data -o ./processed

# Train UNetR model
nifti-model train -d ./processed -o ./models --epochs 100

# Generate predictions
nifti-model predict -m ./models/best.ckpt -i volume.nii.gz -o prediction.nii.gz

# Evaluate model performance  
nifti-model evaluate -m ./models/best.ckpt -t ./test_data
```

## 🏥 Hounsfield Unit Support

The pipeline includes specialized transforms for medical imaging:

- **`HounsfieldUnitNormalizationd`**: Fixed HU windowing with clinical presets
- **`AdaptiveHUNormalizationd`**: Automatic percentile-based windowing
- **Clinical Presets**: soft_tissue, lung, bone, brain, cardiac, etc.

```python
# Configure HU processing
config = Config(
    data=DataConfig(
        use_adaptive_hu_normalization=True,
        hu_window_preset="soft_tissue",  # or custom range
        hu_min=-160.0,
        hu_max=240.0
    )
)
```

## 🔧 Installation

```bash
# Install dependencies
uv add monai torch lightning click loguru nibabel wandb

# Install in development mode
uv pip install -e .
```

## 📝 Configuration

Use YAML configuration files for reproducible experiments:

```yaml
data:
  data_dir: "./data"
  use_adaptive_hu_normalization: true
  target_size: [224, 224, 64]

model:
  img_size: [224, 224, 64]
  feature_size: 16

training:
  epochs: 100
  batch_size: 1
  learning_rate: 1e-4

wandb:
  enabled: true
  project: "medical-segmentation"
```

## 🎯 Key Features

- ✅ **Modular Design**: Logically organized submodules
- ✅ **Medical Imaging**: Hounsfield Unit preprocessing
- ✅ **UNetR Architecture**: MONAI-based implementation
- ✅ **Lightning Integration**: Professional training workflows
- ✅ **Experiment Tracking**: Weights & Biases support
- ✅ **Volume Processing**: Sliding window inference
- ✅ **CLI Interface**: Docker-style commands
- ✅ **Type Safety**: Full type hints throughout