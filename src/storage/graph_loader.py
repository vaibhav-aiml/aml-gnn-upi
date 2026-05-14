"""
Graph loader for loading/saving graphs to Neo4j
"""
import pandas as pd
import numpy as np
from pathlib import Path

class GraphLoader:
    def __init__(self, neo4j_client=None):
        self.neo4j_client = neo4j_client
        
    def save_graph_to_neo4j(self, graph_data, transactions_df):
        """Save graph to Neo4j database"""
        if self.neo4j_client:
            self.neo4j_client.create_transaction_graph(transactions_df)
            print("✅ Graph saved to Neo4j")
        else:
            print("⚠️ Neo4j client not available - saving to file instead")
            self.save_graph_to_file(graph_data)
    
    def save_graph_to_file(self, graph_data, filepath="data/processed/graph_data.pkl"):
        """Save graph to pickle file"""
        import pickle
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"✅ Graph saved to {filepath}")
    
    def load_graph_from_file(self, filepath="data/processed/graph_data.pkl"):
        """Load graph from pickle file"""
        import pickle
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"✅ Graph loaded from {filepath}")
        return graph_data
    
    def load_transactions_from_csv(self, filepath="data/raw/synthetic_upi_transactions.csv"):
        """Load transactions from CSV file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return None
            
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} transactions from {filepath}")
        return df
    
    def export_subgraph(self, account_id, depth=2, filepath=None):
        """Export subgraph around an account"""
        if self.neo4j_client:
            subgraph = self.neo4j_client.get_account_network(account_id, depth)
            if filepath:
                import json
                with open(filepath, 'w') as f:
                    json.dump(subgraph, f, indent=2)
            return subgraph
        else:
            print("⚠️ Neo4j client not available")
            return None
    
    def get_graph_statistics(self, graph_data):
        """Get statistics about the graph"""
        if graph_data is None:
            return {}
            
        stats = {
            'num_nodes': graph_data.get('account', {}).get('x', np.array([])).shape[0] if hasattr(graph_data, 'get') else 0,
            'num_edges': 0,
            'num_features': 0,
            'num_classes': 0
        }
        
        # Try to get edge count
        if hasattr(graph_data, 'edge_index'):
            stats['num_edges'] = graph_data.edge_index.shape[1]
        
        # Try to get feature count
        if hasattr(graph_data, 'x'):
            stats['num_features'] = graph_data.x.shape[1] if len(graph_data.x.shape) > 1 else 1
        
        # Try to get class count
        if hasattr(graph_data, 'y'):
            stats['num_classes'] = len(torch.unique(graph_data.y)) if hasattr(graph_data.y, 'unique') else 2
        
        return stats

# Simple loader function for quick use
def load_graph(filepath="data/processed/graph_data.pkl"):
    """Quick load function"""
    loader = GraphLoader()
    return loader.load_graph_from_file(filepath)

def save_graph(graph_data, filepath="data/processed/graph_data.pkl"):
    """Quick save function"""
    loader = GraphLoader()
    loader.save_graph_to_file(graph_data, filepath)

if __name__ == "__main__":
    print("GraphLoader module ready")
    print("Usage: loader = GraphLoader(neo4j_client)")