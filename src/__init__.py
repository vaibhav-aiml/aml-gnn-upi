"""
AML GNN Project - Root module
Graph Neural Network for Anti-Money Laundering Detection
"""

__version__ = "1.0.0"
__author__ = "AML Team"
__description__ = "GNN-based fraud detection for UPI transactions"

from src import data, models, detection, api, dashboard, explainability

__all__ = [
    'data',
    'models', 
    'detection',
    'api',
    'dashboard',
    'explainability'
]