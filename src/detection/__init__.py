"""
Detection module - Pattern detection and alert generation
"""

from .patterns import PatternDetector
from .inference import RealTimeInference, BatchInference
from .alert_generator import AlertGenerator, EmailNotifier, WebhookNotifier

__all__ = [
    'PatternDetector',
    'RealTimeInference',
    'BatchInference',
    'AlertGenerator',
    'EmailNotifier',
    'WebhookNotifier'
]