"""
Configuration module - Settings and logging
"""

from .settings import settings, MODEL_CONFIG, FRAUD_PATTERNS
from .logging_config import setup_logging, logger

__all__ = [
    'settings',
    'MODEL_CONFIG',
    'FRAUD_PATTERNS',
    'setup_logging',
    'logger'
]