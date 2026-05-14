"""
Unit tests for graph builder module
"""
import pytest
import pandas as pd
import torch
from src.data.graph_builder import TransactionGraphBuilder
from src.data.synthetic_data_generator import SyntheticUPIDataGenerator

class TestGraphBuilder:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data"""
        generator = SyntheticUPIDataGenerator(num_accounts=50, num_transactions=500)
        df = generator.generate_dataset()
        return df
    
    def test_graph_creation(self, sample_data):
        """Test if graph builds successfully"""
        builder = TransactionGraphBuilder(sample_data)
        graph = builder.build_graph()
        
        assert graph is not None
        assert 'account' in graph.node_types
        assert graph['account'].x is not None
        assert graph['account'].x.shape[1] == 9  # 9 features
        
    def test_node_features(self, sample_data):
        """Test node feature extraction"""
        builder = TransactionGraphBuilder(sample_data)
        features = builder._create_node_features()
        
        assert len(features) == len(builder.account_encoder.classes_)
        assert features.shape[1] == 9
        
    def test_edge_features(self, sample_data):
        """Test edge feature extraction"""
        builder = TransactionGraphBuilder(sample_data)
        features = builder._create_edge_features()
        
        assert len(features) == len(sample_data)
        assert features.shape[1] == 5  # 5 edge features
        
    def test_node_labels(self, sample_data):
        """Test node label creation"""
        builder = TransactionGraphBuilder(sample_data)
        labels = builder._create_node_labels()
        
        assert len(labels) == len(builder.account_encoder.classes_)
        assert set(labels) <= {0, 1}  # Binary labels
        
    def test_graph_edge_count(self, sample_data):
        """Test edge count matches transactions"""
        builder = TransactionGraphBuilder(sample_data)
        graph = builder.build_graph()
        
        edge_count = graph['account', 'transacts', 'account'].edge_index.shape[1]
        assert edge_count == len(sample_data)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])