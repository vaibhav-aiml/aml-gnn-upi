"""
Explainability module - Model interpretation and visualization
"""

from .gnn_explainer import GNNExplainer, GraphMaskExplainer
from .visualize import ExplanationVisualizer, GraphVisualizer

# Try to import SHAP silently
try:
    from .shap_features import SHAPFeatureAnalyzer, FeatureAttribution
    __all__ = [
        'GNNExplainer',
        'GraphMaskExplainer',
        'SHAPFeatureAnalyzer',
        'FeatureAttribution',
        'ExplanationVisualizer',
        'GraphVisualizer'
    ]
except ImportError:
    from .shap_features import FeatureAttribution
    __all__ = [
        'GNNExplainer',
        'GraphMaskExplainer',
        'FeatureAttribution',
        'ExplanationVisualizer',
        'GraphVisualizer'
    ]
    # No warning - silent fallback