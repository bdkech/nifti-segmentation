# Configuration Examples for NiftiLearn

This directory contains example configuration files for different use cases and hardware setups.

## Available Configurations

### 🚀 `quick_start.yaml`
**Best for:** First-time users and testing
- Small image sizes (128x128) for fast processing
- Minimal training epochs (20) for quick results
- Simple model architecture
- W&B logging disabled for simplicity

```bash
niftilearn train -c config_examples/quick_start.yaml
```

### 📚 `complete_example.yaml`
**Best for:** Understanding all available options
- Comprehensive documentation of every parameter
- Production-ready defaults
- Detailed comments explaining each option
- All configuration sections included

### 💻 `cpu_training.yaml`
**Best for:** Systems without GPU acceleration
- Optimized for CPU training
- Smaller batch sizes and model
- Full precision (32-bit)
- Reduced complexity for faster CPU processing

```bash
niftilearn train -c config_examples/cpu_training.yaml
```

### 🔥 `multi_gpu_production.yaml`
**Best for:** High-performance training with multiple GPUs
- Large batch sizes (16) for multi-GPU setups
- Higher resolution images (256x256)
- Advanced model architecture
- Learning rate scheduling
- Extended training (200 epochs)

```bash
niftilearn train -c config_examples/multi_gpu_production.yaml
```

### 🍎 `apple_silicon.yaml`
**Best for:** Mac with Apple Silicon (M1/M2/M3) chips
- Metal Performance Shaders (MPS) acceleration
- Optimized for Apple's unified memory
- Full precision (works better with MPS)
- Appropriate batch sizes for Apple Silicon

```bash
niftilearn train -c config_examples/apple_silicon.yaml
```

## Configuration Sections

### Data Configuration
- **data_dir**: Path to NIFTI volume files
- **annotation_dir**: Path to segmentation mask files
- **slice_axis**: Which axis to extract 2D slices from (0=axial, 1=coronal, 2=sagittal)
- **img_size**: Target size for 2D slices [Height, Width]
- **target_spacing**: Voxel spacing in mm [x, y, z]
- **use_adaptive_hu_normalization**: Automatically adjust Hounsfield Unit windowing

### Compute Configuration
- **devices**: GPU configuration ("auto", "cpu", specific GPUs)
- **accelerator**: Hardware type ("gpu", "cpu", "mps")
- **precision**: Training precision ("32", "16-mixed", "bf16-mixed")
- **num_workers**: Parallel data loading workers

### Model Configuration
- **features**: UNet encoder-decoder channel sizes
- **activation**: Activation function ("RELU", "PRELU", "LEAKYRELU")
- **in_channels/out_channels**: Input/output channel counts

### Training Configuration
- **optimizer**: Optimizer type ("Adam", "AdamW", "SGD")
- **loss_function**: Loss function ("dice", "dicece", "focal")
- **scheduler**: Learning rate scheduler (optional)
- **patience**: Early stopping patience

### W&B Configuration
- **enabled**: Enable/disable Weights & Biases logging
- **project**: W&B project name
- **tags**: Experiment tags for organization

## Customizing Configurations

### 1. Start with an appropriate template
Choose the configuration that best matches your hardware and use case.

### 2. Modify key parameters
Common modifications:
- **data_dir/annotation_dir**: Point to your data
- **img_size**: Adjust for your image resolution needs
- **inference_chunk_size**: Increase/decrease based on GPU memory during inference
- **epochs**: Adjust training duration
- **learning_rate**: Tune for your dataset

### 3. Validate your configuration
```bash
python -c "from niftilearn.config.loader import load_config; load_config('your_config.yaml')"
```

### 4. Use CLI overrides for quick testing
```bash
niftilearn train -c config.yaml --epochs 5 --inference-chunk-size 2
```

## Hardware-Specific Tips

### NVIDIA GPUs
- Use `accelerator: "gpu"` and `precision: "16-mixed"`
- Enable multiple GPUs with `devices: [0, 1, 2, 3]`
- Use `strategy: "ddp"` for multi-GPU training

### Apple Silicon
- Use `accelerator: "mps"` for Metal acceleration
- Stick to `precision: "32"` (mixed precision less stable on MPS)
- Single device only: `devices: 1`

### CPU Only
- Use `accelerator: "cpu"` and `precision: "32"`
- Reduce `inference_chunk_size` and `img_size` for reasonable performance
- Consider fewer `num_workers`

## Common Issues

### Memory Errors
- Reduce `inference_chunk_size` and/or `img_size`
- Use `precision: "16-mixed"` on compatible hardware
- Decrease model `features`

### Slow Training
- Increase `num_workers` (but not more than CPU cores)
- Use appropriate `accelerator` for your hardware
- Enable mixed precision training

### Configuration Validation Errors
- Ensure `data.slice_axis == model.slice_axis`
- Ensure `data.img_size == model.img_size`
- Ensure `data.inference_chunk_size == training.inference_chunk_size`
- Verify all splits sum to 1.0

For more help, see the main documentation or create an issue on GitHub.