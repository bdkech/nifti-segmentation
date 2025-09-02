"""Configuration loading utilities for NiftiLearn."""

from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger
from pydantic import ValidationError

from niftilearn.config.models import Config


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""


def load_yaml(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration data

    Raises:
        ConfigurationError: If file cannot be read or parsed
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            raise ConfigurationError(
                f"Configuration file is empty: {config_path}"
            )

        if not isinstance(config_data, dict):
            raise ConfigurationError(
                f"Configuration must be a dictionary, got {type(config_data).__name__}"
            )

        logger.info(f"Loaded configuration from: {config_path}")
        return config_data

    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except PermissionError:
        raise ConfigurationError(
            f"Permission denied reading configuration: {config_path}"
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Unexpected error loading {config_path}: {e}")


def validate_config(config_data: dict[str, Any]) -> Config:
    """Validate configuration data using Pydantic models.

    Args:
        config_data: Raw configuration dictionary

    Returns:
        Validated configuration object

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        config = Config(**config_data)
        logger.info("Configuration validation successful")
        return config

    except ValidationError as e:
        error_msg = "Configuration validation failed:\n"
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            error_msg += f"  {loc}: {error['msg']}\n"
        raise ConfigurationError(error_msg.strip())
    except Exception as e:
        raise ConfigurationError(f"Unexpected validation error: {e}")


def apply_cli_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """Apply command-line overrides to configuration.

    Args:
        config: Base configuration object
        overrides: Dictionary of CLI overrides

    Returns:
        Updated configuration object

    Raises:
        ConfigurationError: If overrides are invalid
    """
    if not overrides:
        return config

    # Convert to dict, apply overrides, and re-validate
    config_dict = config.dict()

    try:
        # Apply training overrides
        if "epochs" in overrides:
            config_dict["training"]["epochs"] = overrides["epochs"]
        if "inference_chunk_size" in overrides:
            config_dict["training"]["inference_chunk_size"] = overrides[
                "inference_chunk_size"
            ]
            config_dict["data"]["inference_chunk_size"] = overrides[
                "inference_chunk_size"
            ]
        if "learning_rate" in overrides:
            config_dict["training"]["learning_rate"] = overrides[
                "learning_rate"
            ]
        if "annotation_type" in overrides:
            config_dict["data"]["annotation_type"] = overrides[
                "annotation_type"
            ]

        # Re-validate with overrides
        updated_config = Config(**config_dict)

        logger.info(f"Applied CLI overrides: {list(overrides.keys())}")
        return updated_config

    except ValidationError as e:
        error_msg = "CLI override validation failed:\n"
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            error_msg += f"  {loc}: {error['msg']}\n"
        raise ConfigurationError(error_msg.strip())
    except Exception as e:
        raise ConfigurationError(f"Unexpected error applying overrides: {e}")


def load_config(
    config_path: Path, cli_overrides: Optional[dict[str, Any]] = None
) -> Config:
    """Load and validate configuration from YAML file with CLI overrides.

    Args:
        config_path: Path to YAML configuration file
        cli_overrides: Optional dictionary of CLI parameter overrides

    Returns:
        Validated configuration object

    Raises:
        ConfigurationError: If loading or validation fails
    """
    logger.info(f"Loading configuration from: {config_path}")

    # Load YAML data
    config_data = load_yaml(config_path)

    # Validate base configuration
    config = validate_config(config_data)

    # Apply CLI overrides if provided
    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {config.output_dir}")

    return config


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save
        output_path: Path for output YAML file

    Raises:
        ConfigurationError: If saving fails
    """
    try:
        # Convert to dictionary and handle Path objects
        config_dict = config.dict()

        # Convert Path objects to strings for YAML serialization
        def path_to_str(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [path_to_str(item) for item in obj]
            else:
                return obj

        config_dict = path_to_str(config_dict)

        # Save to YAML
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {output_path}")

    except Exception as e:
        raise ConfigurationError(
            f"Failed to save configuration to {output_path}: {e}"
        )
