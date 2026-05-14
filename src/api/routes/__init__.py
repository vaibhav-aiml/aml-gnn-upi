"""
API Routes - Endpoint implementations
"""

from .predict import router as predict_router
from .explain import router as explain_router
from .health import router as health_router

__all__ = [
    'predict_router',
    'explain_router',
    'health_router'
]