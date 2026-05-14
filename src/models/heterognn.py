"""
Heterogeneous GNN for account + merchant node types
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('account', 'transacts', 'account'): SAGEConv(hidden_channels, hidden_channels),
                ('account', 'pays', 'merchant'): SAGEConv(hidden_channels, hidden_channels),
                ('merchant', 'receives', 'account'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.3, training=self.training) 
                      for key, x in x_dict.items()}
        
        # Only output account predictions
        return self.lin(x_dict['account'])

class SimpleHeteroGNN(nn.Module):
    """Simpler heterogeneous GNN for faster training"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.conv1 = HeteroConv({
            ('account', 'transacts', 'account'): SAGEConv(in_channels, hidden_channels),
            ('account', 'pays', 'merchant'): SAGEConv(in_channels, hidden_channels),
        })
        
        self.conv2 = HeteroConv({
            ('account', 'transacts', 'account'): SAGEConv(hidden_channels, hidden_channels),
            ('account', 'pays', 'merchant'): SAGEConv(hidden_channels, hidden_channels),
        })
        
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        return self.lin(x_dict['account'])

if __name__ == "__main__":
    model = SimpleHeteroGNN(in_channels=9, hidden_channels=64, out_channels=2)
    print(f"HeteroGNN parameters: {sum(p.numel() for p in model.parameters()):,}")