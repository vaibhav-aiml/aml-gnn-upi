"""
GNNExplainer for interpreting model predictions
Critical for regulatory compliance
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import numpy as np

class GNNExplainer:
    def __init__(self, model, epochs=100, lr=0.01):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        
    def explain_node(self, node_idx, x, edge_index, edge_attr=None):
        """Explain prediction for a specific node"""
        self.model.eval()
        
        # Get subgraph around node
        subset, edge_index_sub, mapping, edge_mask_sub = k_hop_subgraph(
            node_idx, 3, edge_index, relabel_nodes=True
        )
        x_sub = x[subset]
        
        # Initialize edge mask
        edge_mask = torch.ones(edge_index_sub.size(1), requires_grad=True)
        
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply mask to edges
            masked_edge_index = edge_index_sub
            if edge_attr is not None:
                masked_edge_attr = edge_attr[edge_mask_sub] * edge_mask.sigmoid()
            else:
                masked_edge_attr = None
            
            # Forward pass
            logits = self.model(x_sub, masked_edge_index)
            logits = F.softmax(logits, dim=1)
            
            # Get prediction for target node
            node_idx_in_sub = mapping[node_idx]
            pred = logits[node_idx_in_sub]
            
            # Calculate loss (maximize confidence for predicted class)
            loss = -torch.log(pred[pred.argmax()])
            
            loss.backward()
            optimizer.step()
        
        # Get important edges
        edge_importance = edge_mask.sigmoid().detach().numpy()
        
        return {
            'node_idx': node_idx,
            'important_edges': edge_importance,
            'edge_index': edge_index_sub,
            'confidence': pred.max().item()
        }
    
    def explain_subgraph(self, nodes, x, edge_index):
        """Explain predictions for multiple nodes"""
        explanations = []
        for node in nodes:
            explanation = self.explain_node(node, x, edge_index)
            explanations.append(explanation)
        return explanations
    
    def visualize_importance(self, explanation):
        """Get importance scores for visualization"""
        important_edges = explanation['important_edges']
        top_edges = np.argsort(important_edges)[-10:]  # Top 10 important edges
        
        return {
            'top_edges': top_edges,
            'importance_scores': important_edges[top_edges]
        }

class GraphMaskExplainer:
    """Alternative explainer using GraphMask"""
    def __init__(self, model):
        self.model = model
        
    def explain(self, x, edge_index, target_node):
        """Explain using gradient information"""
        self.model.eval()
        
        # Compute gradients
        x.requires_grad = True
        logits = self.model(x, edge_index)
        logits[target_node, 1].backward()
        
        # Get feature importance
        feature_importance = x.grad[target_node].abs().detach().numpy()
        
        return feature_importance

if __name__ == "__main__":
    print("GNNExplainer ready for use")