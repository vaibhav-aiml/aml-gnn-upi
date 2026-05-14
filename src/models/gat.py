"""
Graph Attention Network (GAT) for fraud detection
GAT learns which neighbors are most important for detecting fraud
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATFraudDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1, dropout=dropout)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # First attention layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second attention layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv3(x, edge_index)
        
        return x
    
    def get_attention_weights(self, x, edge_index):
        """Extract attention weights to see which neighbors matter"""
        attention_weights = []
        
        # Get attention from first layer
        attn1 = self.conv1.attentions if hasattr(self.conv1, 'attentions') else None
        if attn1 is not None:
            attention_weights.append(attn1)
        
        return attention_weights

class MultiHeadGAT(nn.Module):
    """Multi-head GAT with skip connections"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

if __name__ == "__main__":
    model = GATFraudDetector(9, 64, 2, heads=4)
    print(f"GAT Model parameters: {sum(p.numel() for p in model.parameters()):,}")