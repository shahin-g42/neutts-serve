"""
Logging configuration for the TTS service.
"""

import sys
import os
from loguru import logger
from core.settings import settings


# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)

# Add file handler
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
logger.add(
    settings.log_file,
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=settings.log_level
)

# Export logger
app_logger = logger
