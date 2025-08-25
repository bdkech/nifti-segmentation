"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_dir: Path = Field(..., description="Directory containing NIFTI volumes")
    annotation_dir: Optional[Path] = Field(
        None, description="Directory containing annotation volumes"
    )
    
    # Data splits
    train_split: float = Field(0.8, ge=0.0, le=1.0, description="Training data fraction")
    val_split: float = Field(0.15, ge=0.0, le=1.0, description="Validation data fraction")
    test_split: float = Field(0.05, ge=0.0, le=1.0, description="Test data fraction")
    
    # File patterns
    volume_pattern: str = Field("*.nii.gz", description="Pattern for volume files")
    annotation_pattern: str = Field("*_seg.nii.gz", description="Pattern for annotation files")
    
    # Processing parameters
    target_spacing: Optional[list[float]] = Field(
        None, description="Target voxel spacing [x, y, z]"
    )
    target_size: Optional[list[int]] = Field(
        None, description="Target volume size [H, W, D]"
    )
    
    # Slice processing parameters
    slice_axis: int = Field(2, description="Axis along which to extract slices (0=axial, 1=coronal, 2=sagittal)")
    batch_size: int = Field(8, description="Number of slices per batch (from same volume)")
    img_size: list[int] = Field([224, 224], description="Target slice size [H, W]")
    
    # Hounsfield Unit normalization parameters
    use_adaptive_hu_normalization: bool = Field(
        True, description="Use adaptive HU normalization based on percentiles"
    )
    hu_window_preset: Optional[str] = Field(
        None, 
        description=(
            "Predefined HU windowing preset. Options: 'soft_tissue', 'lung', "
            "'bone', 'brain', 'mediastinum', 'abdomen', 'liver', 'kidney', "
            "'muscle', 'cardiac'"
        )
    )
    hu_min: float = Field(
        -1000.0, description="Minimum HU value for fixed windowing"
    )
    hu_max: float = Field(
        400.0, description="Maximum HU value for fixed windowing"
    )
    adaptive_hu_lower_percentile: float = Field(
        0.5, ge=0.0, le=50.0, description="Lower percentile for adaptive HU windowing"
    )
    adaptive_hu_upper_percentile: float = Field(
        99.5, ge=50.0, le=100.0, description="Upper percentile for adaptive HU windowing"
    )
    
    @field_validator("test_split")
    @classmethod
    def validate_splits(cls, v: float, info) -> float:
        """Ensure splits sum to 1.0."""
        if info.data:
            train_split = info.data.get("train_split", 0.0)
            val_split = info.data.get("val_split", 0.0)
            total = train_split + val_split + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Data splits must sum to 1.0, got {total}")
        return v


class ModelConfig(BaseModel):
    """Configuration for 2D UNet model architecture."""
    
    # Architecture parameters
    img_size: list[int] = Field(
        [224, 224], description="Input image size [H, W] for 2D slices"
    )
    in_channels: int = Field(1, description="Number of input channels")
    out_channels: int = Field(1, description="Number of output channels")
    
    # UNet specific parameters
    features: tuple[int, ...] = Field(
        (32, 32, 64, 128, 256, 32), 
        description="Feature channels for each UNet level"
    )
    activation: str = Field("PRELU", description="Activation function")
    norm_name: str = Field("INSTANCE", description="Normalization layer type")
    dropout_rate: float = Field(0.0, ge=0.0, le=1.0, description="Dropout rate")
    
    # Training parameters - fixed to 2D for slice processing
    spatial_dims: int = Field(2, description="Spatial dimensions (fixed to 2D)")
    
    # Slice extraction parameters
    slice_axis: int = Field(2, description="Axis along which to extract slices (0=axial, 1=coronal, 2=sagittal)")


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    # Basic training parameters
    epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(8, gt=0, description="Batch size (slices per batch from same volume)")
    learning_rate: float = Field(1e-4, gt=0.0, description="Learning rate")
    weight_decay: float = Field(1e-5, ge=0.0, description="Weight decay")
    
    # Optimization
    optimizer: str = Field("AdamW", description="Optimizer type")
    scheduler: Optional[str] = Field(None, description="Learning rate scheduler")
    grad_clip_val: Optional[float] = Field(None, description="Gradient clipping value")
    
    # Loss function
    loss_function: str = Field("DiceCELoss", description="Loss function name")
    loss_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Loss function parameters"
    )
    
    # Validation
    val_interval: int = Field(1, gt=0, description="Validation frequency (epochs)")
    val_metric: str = Field("val_dice", description="Primary validation metric")
    
    # Checkpointing
    save_top_k: int = Field(3, description="Number of best checkpoints to save")
    checkpoint_every_n_epochs: int = Field(10, description="Checkpoint frequency")
    
    # Early stopping
    patience: Optional[int] = Field(None, description="Early stopping patience")
    min_delta: float = Field(0.0, description="Minimum change for early stopping")


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases logging."""
    
    enabled: bool = Field(True, description="Enable W&B logging")
    project: str = Field("nifti-segmentation", description="W&B project name")
    entity: Optional[str] = Field(None, description="W&B entity/team")
    name: Optional[str] = Field(None, description="Run name")
    tags: list[str] = Field(default_factory=list, description="Run tags")
    notes: Optional[str] = Field(None, description="Run description")
    
    # Logging parameters
    log_model: bool = Field(True, description="Log model artifacts")
    log_gradients: bool = Field(False, description="Log gradients")
    log_parameters: bool = Field(True, description="Log hyperparameters")


class Config(BaseModel):
    """Main configuration class."""
    
    # Sub-configurations
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    
    # Output paths
    output_dir: Path = Field(
        Path("./outputs"), description="Base output directory"
    )
    checkpoint_dir: Optional[Path] = Field(
        None, description="Checkpoint directory (defaults to output_dir/checkpoints)"
    )
    log_dir: Optional[Path] = Field(
        None, description="Log directory (defaults to output_dir/logs)"
    )
    
    # Compute settings
    accelerator: str = Field("auto", description="Training accelerator")
    devices: Optional[int] = Field(None, description="Number of devices to use")
    num_workers: int = Field(4, ge=0, description="Number of data loader workers")
    precision: str = Field("32", description="Training precision")
    
    # Random seed
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    def __post_init__(self):
        """Set default paths after initialization."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / "checkpoints"
        if self.log_dir is None:
            self.log_dir = self.output_dir / "logs"
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            pydantic.ValidationError: If config is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            yaml.dump(
                self.dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )


def create_default_config() -> Config:
    """Create default configuration.
    
    Returns:
        Default Config instance
    """
    return Config(
        data=DataConfig(
            data_dir=Path("./data"),
        ),
        model=ModelConfig(),
        training=TrainingConfig(),
    )