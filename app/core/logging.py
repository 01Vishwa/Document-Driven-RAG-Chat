"""
Aviation RAG Chat - Logging Configuration
Centralized logging using Loguru
"""

import sys
from pathlib import Path
from loguru import logger

from app.core.config import settings


def setup_logging():
    """Configure application logging."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if settings.debug_mode else "INFO",
        colorize=True,
    )
    
    # File handler
    log_dir = settings.base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_dir / "app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )
    
    logger.info("Logging configured successfully")


# Initialize logging on module import
setup_logging()
