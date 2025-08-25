"""Logging configuration using Loguru."""

import sys
from typing import Optional

from loguru import logger


def setup_logging(
    verbose: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging with Loguru.
    
    Args:
        verbose: Enable verbose (INFO level) logging
        debug: Enable debug logging 
        log_file: Optional file path to write logs to
        
    Notes:
        - Default level is WARNING
        - Verbose enables INFO level
        - Debug enables DEBUG level (overrides verbose)
        - Logs are formatted with timestamp, level, module, and message
    """
    # Remove default handler
    logger.remove()
    
    # Determine log level
    if debug:
        level = "DEBUG"
    elif verbose:
        level = "INFO"
    else:
        level = "WARNING"
    
    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # Optional file handler
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            ),
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )
    
    logger.info(f"Logging configured at {level} level")