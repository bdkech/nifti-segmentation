"""Logging infrastructure for NiftiLearn."""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def setup_logging(
    level: str = "INFO", 
    log_file: Optional[Path] = None,
    show_time: bool = True,
    show_level: bool = True,
) -> None:
    """Set up logging configuration for NiftiLearn.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        show_time: Whether to show timestamp in console output
        show_level: Whether to show log level in console output
    """
    # Remove default logger
    logger.remove()
    
    # Console format based on level and options
    console_format = ""
    if show_time:
        console_format += "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    if show_level:
        console_format += "<level>{level: <8}</level> | "
    console_format += "<level>{message}</level>"
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            str(log_file),
            format=file_format,
            level="DEBUG",  # Always log everything to file
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging initialized with level: {level}")


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a Rich progress bar for training/processing.
    
    Args:
        description: Description text for the progress bar
        
    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(show_speed=True),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        transient=True,
    )


def create_simple_progress() -> Progress:
    """Create a simple progress bar without time estimates.
    
    Returns:
        Simple Progress instance
    """
    return Progress(
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=Console(stderr=True),
        transient=True,
    )


class LoggingMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self):
        """Get logger instance for this class."""
        return logger.bind(classname=self.__class__.__name__)
    
    def log_info(self, message: str) -> None:
        """Log info message with class context."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message with class context."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message with class context."""
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message with class context."""
        self.logger.debug(message)
    
    def log_exception(self, message: str) -> None:
        """Log exception with traceback and class context."""
        self.logger.exception(message)


def log_config_summary(config) -> None:
    """Log a summary of the configuration.
    
    Args:
        config: Configuration object to summarize
    """
    logger.info("=== Configuration Summary ===")
    logger.info(f"Data directory: {config.data.data_dir}")
    logger.info(f"Annotation directory: {config.data.annotation_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Slice axis: {config.data.slice_axis}")
    logger.info(f"Image size: {config.model.img_size}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"GPU devices: {config.compute.devices}")
    logger.info(f"Precision: {config.compute.precision}")
    logger.info(f"W&B enabled: {config.wandb.enabled}")
    logger.info("=== End Configuration ===")


def log_system_info() -> None:
    """Log system and environment information."""
    import platform
    import sys
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
    except ImportError:
        logger.warning("PyTorch not available")
    
    try:
        import lightning
        logger.info(f"Lightning version: {lightning.__version__}")
    except ImportError:
        logger.warning("Lightning not available")
    
    logger.info("=== End System Information ===")