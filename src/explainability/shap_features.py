"""
SHAP analysis for node features (Optional - requires shap package)
"""
import torch
import numpy as np
import pandas as pd

# Try to import shap, but don't fail if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Install with: pip install shap")

class SHAPFeatureAnalyzer:
    def __init__(self, model, background_data):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Please run: pip install shap")
        
        self.model = model
        self.background_data = background_data
        
    def explain_prediction(self, x, edge_index, node_idx):
        """Explain prediction using SHAP"""
        self.model.eval()
        
        # Create explainer
        explainer = shap.GradientExplainer(
            (self.model, self.model.conv1),
            self.background_data
        )
        
        # Get SHAP values
        shap_values = explainer.shap_values(x, edge_index)
        
        # Get feature names
        feature_names = [
            'out_degree', 'in_degree', 'avg_send', 'avg_receive',
            'std_send', 'std_receive', 'total_sent', 'total_received',
            'send_ratio'
        ]
        
        # Get importance for specific node
        node_shap = shap_values[node_idx].mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': node_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_top_features(self, x, edge_index, node_idx, top_k=5):
        """Get top K important features"""
        importance_df = self.explain_prediction(x, edge_index, node_idx)
        return importance_df.head(top_k)
    
    def generate_report(self, x, edge_index, node_idx):
        """Generate detailed SHAP report"""
        importance_df = self.explain_prediction(x, edge_index, node_idx)
        
        report = {
            'node_id': node_idx,
            'top_features': importance_df.head(3).to_dict('records'),
            'bottom_features': importance_df.tail(3).to_dict('records'),
            'feature_distribution': importance_df['importance'].describe().to_dict()
        }
        
        return report

class FeatureAttribution:
    """Simple feature attribution without SHAP (always works)"""
    def __init__(self, model):
        self.model = model
        
    def gradient_attribution(self, x, edge_index, node_idx):
        """Use gradients for feature attribution"""
        self.model.eval()
        
        x.requires_grad = True
        logits = self.model(x, edge_index)
        
        # Get gradient for fraud class
        logits[node_idx, 1].backward()
        
        attribution = x.grad[node_idx].abs().detach().numpy()
        
        return attribution
    
    def simple_attribution(self, x, edge_index, node_idx):
        """Simpler attribution without gradients"""
        with torch.no_grad():
            original_output = self.model(x, edge_index)[node_idx]
            
            # Perturb each feature and see impact
            attribution = []
            for feature_idx in range(x.shape[1]):
                x_perturbed = x.clone()
                x_perturbed[node_idx, feature_idx] = 0
                perturbed_output = self.model(x_perturbed, edge_index)[node_idx]
                
                impact = (original_output[1] - perturbed_output[1]).item()
                attribution.append(abs(impact))
            
            return np.array(attribution)

if __name__ == "__main__":
    print(f"SHAP available: {SHAP_AVAILABLE}")