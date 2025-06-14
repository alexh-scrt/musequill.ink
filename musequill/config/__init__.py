"""Configuration management for MuseQuill."""

from .settings import Settings, get_settings
from .logging import setup_logging, get_logger
from .environments import Environment

# Global settings instance
settings = get_settings()

# Setup logging
setup_logging(settings)

__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "setup_logging",
    "get_logger",
    "Environment",
]