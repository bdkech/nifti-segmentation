"""Command line interface submodule.

This submodule provides:
- Click-based CLI with Docker-style commands
- Command implementations for training, prediction, evaluation
"""

from .main import main

__all__ = [
    "main",
]