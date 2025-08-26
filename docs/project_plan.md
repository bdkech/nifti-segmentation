# Nifti volume segmentation decomposition.

## Objective

The purpose of this project is to develop a pipeline and model for processing medical NIFTI volumes for segmentation tasks.  

The concept is this:

- 3D volumes are too large for most architectures.
- We will split the volumes into their slices and annotations.
  - Annotations will also be in NIFTI format.  They are a binary mask.
  - There are multiple annotation types: ART for artery, RA for rectus abdominus muscle, S_FAT for subcutaneous fat.
- Each batch of training will process one volume.
  - This hopefully eliminates any processing issues and helps learn individual features.

We will use a UNet architecture.  The code should:

1. Provide the methods for loading the NIFTIs.
2. Processing them in batches of 1 NIFTI per batch.
3. Provide the logic for running a training loop for the UNetR architecture.
4. Provide the final segmentation as a volume.
5. Arguments should be set through command line arguments, or a config file.

The directory structure of the input is as follows:

```
|-data_dir/
|-subject_id/
|    |-various subdirectory names with study volume/
|     |- *.nii.gz
|- Segmentations.*.nii.gz
```

## Architecture

This is a command line tool that will be used in high performance workflows.  It should be written with object-oriented practices.  It will be invoked at the command line at a task-level.  Interactions with the CLI should be similar to invoking docker commands.

The following python packages should be used:

| package | description |
| --------|------------- |
| MONAI | Used for data loading, transforms, and model architectures|
| PyTorch| Main deep learning library, torch run used for parallel execution |
| Lightning | Framework for training loops and general data processing|
| Click | CLI interface |
| Loguru | Logging and debugging |
| Nibabel| NIFTI volume loading and managing|
| Weights and Biases| Local model performance tracking and MLOps|
