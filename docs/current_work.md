# Task 9: Model Architecture Implementation - Implementation Checklist

## Create UNet2D model class
- [ ] Implement 2D UNet architecture using MONAI
- [ ] Configure model with ModelConfig parameters
- [ ] Support configurable features and activation functions
- [ ] Implement proper input/output channel handling

## PyTorch Lightning Module
- [ ] Create `NiftiSegmentationModule` inheriting from `LightningModule`
- [ ] Implement `__init__` with model, loss, and optimizer configuration
- [ ] Add `forward()` method for inference
- [ ] Implement `training_step()` with loss calculation
- [ ] Implement `validation_step()` with metrics
- [ ] Add `configure_optimizers()` with scheduler support

## Loss functions and metrics
- [ ] Implement configurable loss functions (Dice, DiceCE, Focal)
- [ ] Add segmentation metrics (Dice coefficient, IoU, sensitivity, specificity)
- [ ] Support multi-class segmentation evaluation
- [ ] Add proper metric aggregation across batches

## Model utilities
- [ ] Add model summary and parameter counting
- [ ] Implement model checkpoint loading/saving
- [ ] Add inference utilities for volume reconstruction
- [ ] Create model validation functions

## Integration and exports
- [ ] Export model classes in `models/__init__.py`
- [ ] Update main package exports
- [ ] Add model configuration validation
- [ ] Ensure compatibility with existing data pipeline