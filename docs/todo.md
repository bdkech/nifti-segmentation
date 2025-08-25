# NIFTI Model - TODO List

## 🔧 Core Implementation Tasks

### High Priority
- [ ] **CLI Command Implementation**
  - [ ] Implement `preprocess` command logic in `cli/main.py:preprocess()`
  - [ ] Implement `train` command logic in `cli/main.py:train()`
  - [ ] Implement `predict` command logic in `cli/main.py:predict()`
  - [ ] Implement `evaluate` command logic in `cli/main.py:evaluate()`
  - [ ] Add configuration loading from file in CLI

- [ ] **Data Pipeline Completion**
  - [ ] Implement test dataset creation in `data/loaders.py:setup()`
  - [ ] Implement prediction data loader in `data/loaders.py:predict_dataloader()`
  - [ ] Add data validation and integrity checks in `data/loaders.py:prepare_data()`
  - [ ] Implement volume annotation matching by filename patterns
  - [ ] Add support for multi-modal volumes (T1, T2, FLAIR, etc.)

- [ ] **Model Enhancements**
  - [ ] Add model summary and parameter counting utilities
  - [ ] Implement model export for ONNX/TorchScript
  - [ ] Add support for different UNetR configurations (base, large, huge)
  - [ ] Implement ensemble prediction capabilities

### Medium Priority
- [ ] **Training Improvements**
  - [ ] Add automatic mixed precision (AMP) support
  - [ ] Implement advanced learning rate scheduling
  - [ ] Add cross-validation training support
  - [ ] Implement model pruning and quantization
  - [ ] Add distributed training optimization

- [ ] **Preprocessing Enhancements**
  - [ ] Add intensity histogram matching across volumes

- [ ] **Inference Optimizations**
  - [ ] Add uncertainty quantification

### Low Priority
- [ ] **Visualization & Analysis**
  - [ ] Create 3D volume visualization tools
  - [ ] Implement slice-by-slice prediction visualization
  - [ ] Add training progress visualization
  - [ ] Create model attention map visualization
  - [ ] Implement prediction uncertainty visualization

- [ ] **Experiment Tracking**
  - [ ] Complete W&B prediction visualization in `training/logging.py:log_predictions()`
  - [ ] Add automatic hyperparameter optimization with W&B Sweeps
  - [ ] Implement model comparison dashboards
  - [ ] Add automatic experiment reporting

## 🧪 Testing & Validation

### Test Suite Creation
- [ ] **Unit Tests**
  - [ ] Test Hounsfield Unit normalization transforms
  - [ ] Test data loading with various NIFTI formats
  - [ ] Test model forward/backward passes
  - [ ] Test configuration validation
  - [ ] Test CLI command parsing

## 📚 Documentation

### User Documentation
- [ ] **Getting Started Guide**
  - [ ] Installation instructions
  - [ ] Data preparation tutorial
  - [ ] First training example
  - [ ] Prediction workflow tutorial

- [ ] **Configuration Guide**
  - [ ] Complete configuration reference
  - [ ] Hounsfield Unit windowing guide
  - [ ] Model architecture options
  - [ ] Training best practices

## 🔬 Research & Development

### Model Improvements
- [ ] **Architecture Exploration**
  - [ ] Compare UNetR vs other architectures (nnUNet, SwinUNETR)

- [ ] **Data Augmentation**
  - [ ] Add domain-specific augmentations
  - [ ] Implement elastic deformations
  - [ ] Add noise and artifact simulation

## 🛠️ DevOps & Deployment

### Deployment
- [ ] **Containerization**
  - [ ] Create production Docker images
  - [ ] Optimize for GPU inference
  - [ ] Multi-stage builds for size optimization

- [ ] **Model Serving**
  - [ ] Implement REST API for predictions
  - [ ] Add model versioning support
  - [ ] Batch prediction endpoint
  - [ ] Real-time inference optimization

## 🐛 Known Issues & Fixes

### Current Issues
- [ ] Fix lambda transforms in preprocessing pipeline (line 481 in transforms.py)

### Error Handling
- [ ] Add comprehensive exception handling throughout

## 🔄 Maintenance Tasks

### Regular Updates
- [ ] **Dependency Management**
  - [ ] Keep MONAI, PyTorch, Lightning up to date
  - [ ] Monitor for security vulnerabilities
  - [ ] Test compatibility with new versions
  - [ ] Update documentation for breaking changes

### Code Refactoring
- [ ] Optimize data loading pipelines
- [ ] Reduce code duplication across modules
- [ ] Improve configuration validation
- [ ] Enhance logging and debugging capabilities

---

## 📝 Notes

- **Priority Levels**: Focus on High Priority items first for MVP
- **Dependencies**: Some items depend on others - check prerequisites
- **Testing**: All new features should include corresponding tests
- **Documentation**: Update docs for any public API changes

**Last Updated**: 2025-01-27
**Version**: 0.1.0