"""
Unit tests for GNN models
"""
import pytest
import torch
from src.models.graphsage import GraphSAGELight, GraphSAGEFraudDetector
from src.models.gat import GATFraudDetector
from src.models.loss import FocalLoss, WeightedBinaryCrossEntropy

class TestModels:
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy graph data"""
        num_nodes = 100
        num_features = 9
        
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, 500))
        y = torch.randint(0, 2, (num_nodes,))
        
        return x, edge_index, y
    
    def test_graphsage_forward(self, dummy_data):
        """Test GraphSAGE forward pass"""
        x, edge_index, y = dummy_data
        
        model = GraphSAGELight(
            in_channels=9,
            hidden_channels=64,
            out_channels=2
        )
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 2)
        assert not torch.isnan(output).any()
        
    def test_gat_forward(self, dummy_data):
        """Test GAT forward pass"""
        x, edge_index, y = dummy_data
        
        model = GATFraudDetector(
            in_channels=9,
            hidden_channels=64,
            out_channels=2,
            heads=4
        )
        
        output = model(x, edge_index)
        
        assert output.shape == (100, 2)
        
    def test_focal_loss(self):
        """Test focal loss computation"""
        predictions = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))
        
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_weighted_bce(self):
        """Test weighted BCE loss"""
        predictions = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))
        
        loss_fn = WeightedBinaryCrossEntropy(pos_weight=10.0)
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
        
    def test_model_parameters(self, dummy_data):
        """Test model parameter count"""
        x, edge_index, y = dummy_data
        
        model = GraphSAGEFraudDetector(9, 64, 2, num_layers=3)
        param_count = sum(p.numel() for p in model.parameters())
        
        assert param_count > 0
        print(f"Model parameters: {param_count:,}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])