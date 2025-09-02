"""PyTorch Lightning DataModule for NIFTI volume processing."""

from typing import Any, Optional

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader

from niftilearn.config.models import ComputeConfig, DataConfig
from niftilearn.data.datasets import (
    VolumeDataBatch,
    VolumeSliceDataset,
    create_datasets_from_discovery,
    volume_batch_collate,
)
from niftilearn.data.loaders import DiscoveryResult, discover_nifti_data


class NiftiDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for NIFTI volume processing.

    This DataModule handles the complete data pipeline from discovery to batching,
    implementing the "one volume per batch" constraint for medical image processing.
    """

    def __init__(
        self,
        data_config: DataConfig,
        compute_config: ComputeConfig,
        enable_caching: bool = False,
        cache_size: int = 10,
        random_seed: int = 42,
    ):
        """Initialize the NIFTI DataModule.

        Args:
            data_config: Data processing configuration (includes annotation_type)
            compute_config: Compute configuration including num_workers
            enable_caching: Enable volume caching in datasets
            cache_size: Maximum number of volumes to cache per dataset
            random_seed: Random seed for reproducible splits

        Raises:
            ValueError: If data directory is invalid or annotation type is unsupported
        """
        super().__init__()

        # Store configuration
        self.data_config = data_config
        self.compute_config = compute_config
        self.annotation_type = data_config.annotation_type
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.random_seed = random_seed

        # Initialize state
        self.discovery_result: Optional[DiscoveryResult] = None
        self.datasets: dict[str, VolumeSliceDataset] = {}
        self.setup_complete = False

        # Validate data directory upfront
        if not data_config.data_dir.exists():
            raise ValueError(
                f"Data directory does not exist: {data_config.data_dir}"
            )

        if not data_config.data_dir.is_dir():
            raise ValueError(
                f"Data path is not a directory: {data_config.data_dir}"
            )

        logger.info(
            f"NiftiDataModule initialized for annotation type: {self.annotation_type}"
        )
        logger.info(f"Data directory: {data_config.data_dir}")
        logger.info(
            f"Caching enabled: {enable_caching}, Cache size: {cache_size}"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for the specified stage.

        Args:
            stage: Training stage ('fit', 'validate', 'test', 'predict', or None for all)

        Raises:
            RuntimeError: If dataset discovery or creation fails
            ValueError: If no volumes are found for the specified annotation type
        """
        if self.setup_complete and stage is None:
            logger.debug("Setup already complete, skipping")
            return

        logger.info(f"Setting up NiftiDataModule (stage: {stage})")

        try:
            # Perform data discovery if not already done
            if self.discovery_result is None:
                logger.info("Starting data discovery...")
                self.discovery_result = discover_nifti_data(self.data_config)

                if not self.discovery_result.pairs:
                    raise RuntimeError(
                        "No valid volume-segmentation pairs found during discovery"
                    )

                # Validate annotation type is available
                if (
                    self.annotation_type
                    not in self.discovery_result.annotation_types
                ):
                    available_types = sorted(
                        self.discovery_result.annotation_types
                    )
                    raise ValueError(
                        f"Annotation type '{self.annotation_type}' not found in data. "
                        f"Available types: {available_types}"
                    )

                # Log discovery summary
                subjects_with_annotation = len(
                    self.discovery_result.get_subjects_with_annotation(
                        self.annotation_type
                    )
                )
                logger.info(
                    f"Found {subjects_with_annotation} subjects with {self.annotation_type} annotations"
                )

            # Create datasets for the specified stage(s)
            if stage in (None, "fit", "validate"):
                if not self.datasets:
                    logger.info("Creating datasets...")
                    self.datasets = create_datasets_from_discovery(
                        discovery_result=self.discovery_result,
                        annotation_type=self.annotation_type,
                        data_config=self.data_config,
                        random_seed=self.random_seed,
                        enable_caching=self.enable_caching,
                        cache_size=self.cache_size,
                    )

                    # Log dataset information
                    for split_name, dataset in self.datasets.items():
                        logger.info(f"  {split_name}: {len(dataset)} volumes")

            # Mark setup as complete
            self.setup_complete = True
            logger.info("NiftiDataModule setup completed successfully")

        except Exception as e:
            logger.exception("Failed to setup NiftiDataModule")
            raise RuntimeError(f"DataModule setup failed: {e}") from e

    def train_dataloader(self) -> DataLoader[VolumeDataBatch]:
        """Create training DataLoader.

        Returns:
            DataLoader configured for volume-based training

        Raises:
            RuntimeError: If setup has not been called or train dataset is missing
        """
        if not self.setup_complete:
            raise RuntimeError("Must call setup() before creating dataloaders")

        if "train" not in self.datasets:
            raise RuntimeError(
                "Train dataset not available - check data splits"
            )

        train_dataset = self.datasets["train"]

        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=1,  # One volume per batch
            shuffle=True,  # Shuffle volumes
            num_workers=self.compute_config.num_workers,
            collate_fn=volume_batch_collate,
            pin_memory=True,
            persistent_workers=self.compute_config.num_workers > 0,
        )

        logger.debug(
            f"Created train DataLoader: {len(train_dataset)} volumes, "
            f"{self.compute_config.num_workers} workers"
        )

        return dataloader

    def val_dataloader(self) -> DataLoader[VolumeDataBatch]:
        """Create validation DataLoader.

        Returns:
            DataLoader configured for volume-based validation

        Raises:
            RuntimeError: If setup has not been called or val dataset is missing
        """
        if not self.setup_complete:
            raise RuntimeError("Must call setup() before creating dataloaders")

        if "val" not in self.datasets:
            raise RuntimeError(
                "Validation dataset not available - check data splits"
            )

        val_dataset = self.datasets["val"]

        dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,  # One volume per batch
            shuffle=False,  # No shuffling for validation
            num_workers=self.compute_config.num_workers,
            collate_fn=volume_batch_collate,
            pin_memory=True,
            persistent_workers=self.compute_config.num_workers > 0,
        )

        logger.debug(
            f"Created val DataLoader: {len(val_dataset)} volumes, "
            f"{self.compute_config.num_workers} workers"
        )

        return dataloader

    def test_dataloader(self) -> Optional[DataLoader[VolumeDataBatch]]:
        """Create test DataLoader.

        Returns:
            DataLoader configured for volume-based testing, or None if no test data

        Raises:
            RuntimeError: If setup has not been called
        """
        if not self.setup_complete:
            raise RuntimeError("Must call setup() before creating dataloaders")

        if "test" not in self.datasets:
            logger.warning("Test dataset not available - returning None")
            return None

        test_dataset = self.datasets["test"]

        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,  # One volume per batch
            shuffle=False,  # No shuffling for testing
            num_workers=self.compute_config.num_workers,
            collate_fn=volume_batch_collate,
            pin_memory=True,
            persistent_workers=self.compute_config.num_workers > 0,
        )

        logger.debug(
            f"Created test DataLoader: {len(test_dataset)} volumes, "
            f"{self.compute_config.num_workers} workers"
        )

        return dataloader

    def predict_dataloader(self) -> DataLoader[VolumeDataBatch]:
        """Create prediction DataLoader.

        Returns:
            DataLoader configured for volume-based prediction (uses test set)

        Raises:
            RuntimeError: If setup has not been called or no data is available
        """
        if not self.setup_complete:
            raise RuntimeError("Must call setup() before creating dataloaders")

        # Use test dataset for predictions, fall back to validation if test not available
        predict_dataset = None
        if "test" in self.datasets:
            predict_dataset = self.datasets["test"]
            logger.debug("Using test dataset for predictions")
        elif "val" in self.datasets:
            predict_dataset = self.datasets["val"]
            logger.debug(
                "Using validation dataset for predictions (no test set)"
            )
        else:
            raise RuntimeError("No dataset available for predictions")

        dataloader = DataLoader(
            dataset=predict_dataset,
            batch_size=1,  # One volume per batch
            shuffle=False,  # No shuffling for predictions
            num_workers=self.compute_config.num_workers,
            collate_fn=volume_batch_collate,
            pin_memory=True,
            persistent_workers=self.compute_config.num_workers > 0,
        )

        logger.debug(
            f"Created predict DataLoader: {len(predict_dataset)} volumes, "
            f"{self.compute_config.num_workers} workers"
        )

        return dataloader

    def get_discovery_summary(self) -> Optional[dict[str, Any]]:
        """Get data discovery summary information.

        Returns:
            Dictionary with discovery statistics, or None if discovery not performed

        Raises:
            RuntimeError: If setup has not been called
        """
        if not self.setup_complete:
            raise RuntimeError(
                "Must call setup() before accessing discovery information"
            )

        if self.discovery_result is None:
            return None

        return {
            "total_subjects": len(self.discovery_result.subjects),
            "total_pairs": len(self.discovery_result.pairs),
            "annotation_types": sorted(self.discovery_result.annotation_types),
            "subjects_with_target_annotation": len(
                self.discovery_result.get_subjects_with_annotation(
                    self.annotation_type
                )
            ),
            "discovery_stats": self.discovery_result.discovery_stats,
            "warnings_count": len(self.discovery_result.warnings),
            "errors_count": len(self.discovery_result.errors),
        }

    def get_dataset_info(self) -> dict[str, dict[str, Any]]:
        """Get information about created datasets.

        Returns:
            Dictionary with dataset information for each split

        Raises:
            RuntimeError: If setup has not been called
        """
        if not self.setup_complete:
            raise RuntimeError(
                "Must call setup() before accessing dataset information"
            )

        dataset_info: dict[str, dict[str, Any]] = {}
        for split_name, dataset in self.datasets.items():
            dataset_info[split_name] = {
                "num_volumes": len(dataset),
                "annotation_type": dataset.annotation_type,
                "cache_enabled": dataset.enable_caching,
                "cache_stats": dataset.get_cache_stats(),
            }

        return dataset_info

    def clear_cache(self) -> None:
        """Clear volume cache for all datasets.

        Raises:
            RuntimeError: If setup has not been called
        """
        if not self.setup_complete:
            logger.warning("DataModule not set up, nothing to clear")
            return

        for dataset in self.datasets.values():
            dataset.clear_cache()

        logger.info("Cleared cache for all datasets")

    def validate_configuration(self) -> dict[str, Any]:
        """Validate DataModule configuration consistency.

        Returns:
            Dictionary with validation results and recommendations

        Raises:
            RuntimeError: If setup has not been called
        """
        if not self.setup_complete:
            raise RuntimeError(
                "Must call setup() before validating configuration"
            )

        validation_results: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "recommendations": [],
            "configuration": {
                "annotation_type": self.annotation_type,
                "num_workers": self.compute_config.num_workers,
                "caching_enabled": self.enable_caching,
                "cache_size": self.cache_size,
            },
        }

        try:
            # Check for empty datasets
            empty_splits = [
                name
                for name, dataset in self.datasets.items()
                if len(dataset) == 0
            ]
            if empty_splits:
                validation_results["issues"].append(
                    f"Empty dataset splits: {empty_splits}"
                )
                validation_results["valid"] = False

            # Check worker configuration
            if self.compute_config.num_workers > 0:
                validation_results["recommendations"].append(
                    "Multi-worker data loading enabled - ensure sufficient system resources"
                )

            # Check caching configuration
            if self.enable_caching and self.cache_size < 5:
                validation_results["recommendations"].append(
                    "Small cache size may limit performance benefits"
                )

            logger.info("Configuration validation completed")

        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Validation error: {e}")
            logger.exception("Configuration validation failed")

        return validation_results
