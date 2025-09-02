"""Pydantic configuration models for NiftiLearn."""

from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Data processing configuration."""

    data_dir: Path = Field(..., description="Path to data directory")
    annotation_dir: Path = Field(
        ..., description="Path to annotation directory"
    )
    annotation_type: Literal["ART", "RA", "S_FAT"] = Field(
        "ART", description="Target annotation type to load"
    )
    train_split: float = Field(
        0.7, ge=0.0, le=1.0, description="Training split ratio"
    )
    val_split: float = Field(
        0.2, ge=0.0, le=1.0, description="Validation split ratio"
    )
    test_split: float = Field(
        0.1, ge=0.0, le=1.0, description="Test split ratio"
    )

    # Slice processing parameters
    slice_axis: Literal[0, 1, 2] = Field(
        2, description="Axis for slice extraction"
    )
    inference_chunk_size: int = Field(
        8,
        gt=0,
        description="Number of slices to process at once during inference",
    )
    img_size: list[int] = Field(
        [224, 224], description="Target 2D slice size [H, W]"
    )

    # Volume processing parameters
    target_spacing: list[float] = Field(
        [1.0, 1.0, 1.0], description="Target voxel spacing [x, y, z]"
    )
    target_size: list[int] = Field(
        [224, 224, 64],
        description="Target volume size before slicing [H, W, D]",
    )

    # Hounsfield Unit normalization
    use_adaptive_hu_normalization: bool = Field(
        True, description="Use adaptive HU normalization based on percentiles"
    )
    adaptive_hu_lower_percentile: float = Field(
        0.5,
        ge=0.0,
        le=100.0,
        description="Lower percentile for adaptive HU windowing",
    )
    adaptive_hu_upper_percentile: float = Field(
        99.5,
        ge=0.0,
        le=100.0,
        description="Upper percentile for adaptive HU windowing",
    )

    # Fixed HU windowing (when adaptive is disabled)
    hu_window_preset: Optional[Literal["soft_tissue", "bone", "lung"]] = Field(
        None, description="Predefined HU window preset"
    )
    hu_min: Optional[float] = Field(None, description="Minimum HU value")
    hu_max: Optional[float] = Field(None, description="Maximum HU value")

    @model_validator(mode="after")
    def validate_splits(self) -> "DataConfig":
        """Ensure splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")
        return self

    @field_validator("img_size")
    @classmethod
    def validate_img_size(cls, v: list[int]) -> list[int]:
        """Validate img_size has exactly 2 elements."""
        if len(v) != 2:
            raise ValueError("img_size must have exactly 2 elements [H, W]")
        return v

    @field_validator("target_spacing", "target_size")
    @classmethod
    def validate_3d_lists(
        cls, v: list[Union[int, float]]
    ) -> list[Union[int, float]]:
        """Validate 3D lists have exactly 3 elements."""
        if len(v) != 3:
            raise ValueError(
                "target_spacing and target_size must have exactly 3 elements [x, y, z]"
            )
        return v


class ComputeConfig(BaseModel):
    """Compute and GPU configuration."""

    devices: Union[str, int, list[int]] = Field(
        "auto",
        description="GPU devices to use: 'auto', 'cpu', int, or list of ints",
    )
    accelerator: Literal["gpu", "cpu", "mps"] = Field(
        "gpu", description="Accelerator type"
    )
    strategy: Literal["ddp", "fsdp", "auto"] = Field(
        "ddp", description="Multi-GPU strategy"
    )
    precision: Literal["32", "16-mixed", "bf16-mixed"] = Field(
        "16-mixed", description="Training precision"
    )
    num_workers: int = Field(
        4, ge=0, description="Number of data loader workers"
    )


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    img_size: list[int] = Field(
        [224, 224], description="Input image size [H, W]"
    )
    in_channels: int = Field(1, gt=0, description="Number of input channels")
    out_channels: int = Field(1, gt=0, description="Number of output channels")
    features: list[int] = Field(
        [32, 32, 64, 128, 256, 32],
        description="UNet feature channels for each level",
    )
    activation: Literal["RELU", "PRELU", "LEAKYRELU"] = Field(
        "PRELU", description="Activation function"
    )
    spatial_dims: Literal[2] = Field(
        2, description="Spatial dimensions (fixed to 2D for slice processing)"
    )
    slice_axis: Literal[0, 1, 2] = Field(
        2, description="Slice axis (must match data.slice_axis)"
    )

    @field_validator("img_size")
    @classmethod
    def validate_model_img_size(cls, v: list[int]) -> list[int]:
        """Validate image size."""
        if len(v) != 2:
            raise ValueError("img_size must have exactly 2 elements [H, W]")
        if any(x <= 0 for x in v):
            raise ValueError("Image size values must be positive")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(100, gt=0, description="Number of training epochs")
    inference_chunk_size: int = Field(
        8,
        gt=0,
        description="Number of slices to process at once during inference",
    )
    learning_rate: float = Field(
        1e-4, gt=0, description="Initial learning rate"
    )
    optimizer: Literal["Adam", "AdamW", "SGD"] = Field(
        "AdamW", description="Optimizer type"
    )
    loss_function: Literal["dice", "dicece", "focal"] = Field(
        "dicece", description="Loss function (matches training module names)"
    )
    loss_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional loss function parameters"
    )
    scheduler: Optional[
        Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    ] = Field(None, description="Learning rate scheduler type")
    scheduler_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional scheduler parameters"
    )
    patience: int = Field(10, gt=0, description="Early stopping patience")
    min_delta: float = Field(
        1e-4, ge=0.0, description="Minimum change for improvement"
    )


class WandBConfig(BaseModel):
    """Weights and Biases configuration."""

    enabled: bool = Field(True, description="Enable W&B logging")
    project: str = Field("niftilearn", description="W&B project name")
    tags: list[str] = Field(
        default_factory=lambda: ["unet2d", "slice-based"],
        description="Experiment tags",
    )
    name: Optional[str] = Field(None, description="Experiment name")
    notes: Optional[str] = Field(None, description="Experiment notes")


class Config(BaseModel):
    """Root configuration model."""

    data: DataConfig
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    model: ModelConfig
    training: TrainingConfig
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    output_dir: Path = Field(Path("./outputs"), description="Output directory")

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory is absolute."""
        return v.resolve()

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Ensure slice_axis matches between data and model
        if self.data.slice_axis != self.model.slice_axis:
            raise ValueError(
                f"data.slice_axis ({self.data.slice_axis}) must match "
                f"model.slice_axis ({self.model.slice_axis})"
            )

        # Ensure img_size matches between data and model
        if self.data.img_size != self.model.img_size:
            raise ValueError(
                f"data.img_size ({self.data.img_size}) must match "
                f"model.img_size ({self.model.img_size})"
            )

        # Ensure inference_chunk_size matches between data and training
        if self.data.inference_chunk_size != self.training.inference_chunk_size:
            raise ValueError(
                f"data.inference_chunk_size ({self.data.inference_chunk_size}) must match "
                f"training.inference_chunk_size ({self.training.inference_chunk_size})"
            )
