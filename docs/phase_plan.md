# NiftiLearn Development Phases

## Current Phase: Phase 2 - Data Loading & Processing (Tasks 5-7 ✅ | Task 8 Remaining)

### Task 5: NIFTI Data Discovery & Loading (`niftilearn/data/loaders.py`) ✅ COMPLETED
- [x] **Data discovery system**: Scan directory structure to find volumes and matching segmentations
  - [x] Handle `data_dir/subject_id/study_volume/*.nii.gz` structure  
  - [x] Match segmentation files `Segmentations.*.nii.gz` with annotation types
  - [x] Support multiple annotation types: ART (artery), RA (rectus abdominus), S_FAT (subcutaneous fat)
- [x] **Volume loading utilities**: Use nibabel + MONAI for robust NIFTI loading
  - [x] Validate volume-segmentation pairs and consistency
  - [x] Extract metadata using existing `niftilearn.core.metadata` utilities
  - [x] Handle different orientations and spacing automatically

**Implementation Details:**
- Created `VolumeSegmentationPair` and `DiscoveryResult` dataclasses
- Implemented `NiftiDataDiscovery` class with comprehensive file discovery
- Added spatial consistency validation between volumes and segmentations
- Integrated with existing configuration and metadata systems
- Comprehensive error handling and logging throughout discovery process

### Task 6: 2D Slice Extraction System (`niftilearn/data/transforms.py`) ✅ COMPLETED
- [x] **3D to 2D slice conversion**: Extract slices along specified axis (axial/coronal/sagittal)
  - [x] Maintain spatial relationships between volume and segmentation slices
  - [x] Support configurable slice axis from config (simplified - no empty slice handling needed)
  - [x] PyTorch/Lightning compatible slice extraction with proper tensor shapes
- [x] **Volume preprocessing pipeline**: 
  - [x] Resample to target spacing and size using MONAI transforms
  - [x] Hounsfield Unit normalization (adaptive percentile-based or fixed windowing)
  - [x] Intensity scaling and clipping for robust training
  - [x] VolumeSliceExtractor class for coordinated preprocessing and slicing

**Implementation Details:**
- Created comprehensive MONAI-based preprocessing pipeline with configurable HU windowing
- Implemented `extract_slices()` function for 3D→2D conversion along configurable axis
- Built `VolumeSliceExtractor` class coordinating volume processing and slice extraction  
- Updated Pydantic validators to v2 syntax for configuration compatibility
- Designed for "one volume per batch" constraint with multi-annotation support (ART, RA, S_FAT)
- Memory-efficient processing with robust error handling and logging throughout

### Task 7: PyTorch Dataset Classes (`niftilearn/data/datasets.py`) ✅ COMPLETED
- [x] **VolumeSliceDataset**: Main dataset class for slice-based training
  - [x] One volume per batch processing (batch_size = number of slices from same volume)
  - [x] Support train/validation/test splits with proper volume-level separation
  - [x] Lazy loading with caching for memory efficiency
  - [x] LRU cache with configurable size and eviction policies
- [x] **Data loading strategy**:
  - [x] Volume-level sampling to ensure batch coherency
  - [x] Custom collate function for PyTorch DataLoader compatibility
  - [x] Support for multiple annotation types in single dataset
  - [x] Integration with existing VolumeSliceExtractor and data discovery

**Implementation Details:**
- Created `VolumeSliceDataset` with "one volume per batch" constraint implementation
- Implemented `create_volume_level_splits()` for proper train/val/test separation preventing data leakage
- Built `volume_batch_collate()` custom collation function for PyTorch DataLoader integration
- Added comprehensive caching system with LRU eviction and cache statistics
- Created `create_datasets_from_discovery()` for seamless integration with discovery system
- Included extensive error handling, validation, and logging throughout dataset operations
- Full PyTorch compatibility with proper tensor shapes and DataLoader integration

### Task 8: Data Pipeline Integration (`niftilearn/data/__init__.py`)
- [ ] **DataModule creation**: Lightning DataModule for seamless training integration
  - [ ] Automatic dataset discovery and validation
  - [ ] Train/val/test split management with proper logging
  - [ ] Multi-worker data loading configuration
- [ ] **Pipeline validation**:
  - [ ] Data consistency checks using existing metadata utilities  
  - [ ] Memory usage optimization and performance profiling
  - [ ] Integration with configuration system and CLI

## Remaining Phases

### Phase 3: Model & Training (Priority: High)
- [ ] **Task 9**: Implement 2D UNet model using MONAI components
  - [ ] Configure UNet architecture with specified feature channels
  - [ ] Support configurable activation functions and spatial dimensions
  - [ ] Add model validation and parameter counting
- [ ] **Task 10**: Create Lightning training module with loss functions, metrics, optimizers
  - [ ] Implement DiceLoss, DiceCELoss, and FocalLoss options
  - [ ] Add Dice coefficient and IoU metrics for segmentation
  - [ ] Configure Adam, AdamW, and SGD optimizers with scheduling
- [ ] **Task 11**: Build trainer orchestration with checkpointing and validation
  - [ ] Lightning Trainer setup with multi-GPU support
  - [ ] Model checkpointing and early stopping
  - [ ] Validation loop with metrics logging
- [ ] **Task 12**: Add W&B integration for experiment tracking
  - [ ] Experiment initialization and configuration logging
  - [ ] Training metrics and loss curves
  - [ ] Model artifacts and checkpoint management

### Phase 4: CLI Integration (Priority: High)
- [ ] **Task 13**: Implement functional `train` command with config loading
  - [ ] Integration with data pipeline and training modules
  - [ ] Progress reporting and training status updates
  - [ ] Error handling and recovery mechanisms
- [ ] **Task 14**: Add `predict` command for inference on new volumes
  - [ ] Model loading from checkpoints
  - [ ] Single volume inference pipeline
  - [ ] Batch prediction for multiple volumes
- [ ] **Task 15**: Create volume reconstruction (reassemble 2D predictions to 3D)
  - [ ] Slice-to-volume reconstruction with proper spacing
  - [ ] Handle different slice axes and orientations
  - [ ] Output format matching input NIFTI structure
- [ ] **Task 16**: Add `validate` command for model evaluation
  - [ ] Test dataset evaluation with comprehensive metrics
  - [ ] Statistical analysis and reporting
  - [ ] Visualization and results export

### Phase 5: Testing & Polish (Priority: Medium)
- [ ] **Task 17**: Write unit tests for core components
  - [ ] Data loading and preprocessing tests
  - [ ] Configuration validation tests
  - [ ] Model architecture and training tests
- [ ] **Task 18**: Add integration tests for end-to-end pipeline
  - [ ] Full training pipeline tests with synthetic data
  - [ ] CLI command integration tests
  - [ ] Multi-GPU and distributed training tests
- [ ] **Task 19**: Implement comprehensive error handling
  - [ ] Data validation and error recovery
  - [ ] Training interruption and resume capabilities
  - [ ] Resource management and cleanup
- [ ] **Task 20**: Add CLI help documentation and usage examples
  - [ ] Comprehensive help text and usage examples
  - [ ] Configuration file templates and guides
  - [ ] Troubleshooting documentation

## Key Design Considerations

### Phase 2 Focus Areas (Tasks 5-7 ✅ Complete | Task 8 Remaining)

**Completed:**
1. ✅ **Data Discovery & Loading**: Robust NIFTI file discovery with spatial consistency validation
2. ✅ **Volume Preprocessing**: MONAI-based pipeline with configurable HU windowing and resampling  
3. ✅ **Slice Extraction**: 3D→2D conversion with PyTorch/Lightning compatibility
4. ✅ **Multi-Annotation Support**: Handle ART, RA, S_FAT annotation types with flexible configuration
5. ✅ **PyTorch Dataset Integration**: VolumeSliceDataset with "one volume per batch" processing
6. ✅ **Memory Efficiency**: Lazy loading with intelligent LRU caching system for large medical volumes
7. ✅ **Volume-Level Data Splits**: Proper train/val/test separation preventing data leakage

**Final Phase 2 Task (Task 8):**
1. **Lightning DataModule**: Seamless training integration with automatic dataset discovery
2. **Multi-Worker DataLoading**: Configurable parallel processing with proper collation
3. **Data Pipeline Validation**: End-to-end testing and performance optimization
4. **CLI Integration**: Integration with configuration system and command-line interface

### Overall Project Goals
- Command-line tool for high-performance medical image segmentation workflows
- Object-oriented design with clean separation of concerns
- Docker-like CLI experience with comprehensive configuration support
- Production-ready error handling and logging
- Multi-GPU training support for scalability