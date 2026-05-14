"""
Build PyG graph from transaction data
"""
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

class TransactionGraphBuilder:
    def __init__(self, transaction_df):
        self.df = transaction_df
        self.account_encoder = LabelEncoder()
        
    def build_graph(self):
        """Build heterogeneous graph from transactions"""
        data = HeteroData()
        
        # Encode accounts as node IDs
        all_accounts = pd.concat([self.df['from_account'], self.df['to_account']]).unique()
        account_ids = self.account_encoder.fit_transform(all_accounts)
        
        # Create node features (account-based features)
        node_features = self._create_node_features()
        data['account'].x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edges (transactions)
        from_ids = self.account_encoder.transform(self.df['from_account'])
        to_ids = self.account_encoder.transform(self.df['to_account'])
        
        data['account', 'transacts', 'account'].edge_index = torch.tensor(
            np.array([from_ids, to_ids]), dtype=torch.long
        )
        
        # Add edge features
        edge_features = self._create_edge_features()
        data['account', 'transacts', 'account'].edge_attr = torch.tensor(
            edge_features, dtype=torch.float
        )
        
        # Add labels
        data['account'].y = torch.tensor(
            self._create_node_labels(), dtype=torch.long
        )
        
        return data
    
    def _create_node_features(self):
        """Create node-level features"""
        node_features = []
        accounts = self.account_encoder.classes_
        
        for account in accounts:
            # Transaction history for this account
            as_sender = self.df[self.df['from_account'] == account]
            as_receiver = self.df[self.df['to_account'] == account]
            
            features = [
                len(as_sender),  # out degree
                len(as_receiver),  # in degree
                as_sender['amount'].mean() if len(as_sender) > 0 else 0,  # avg send amount
                as_receiver['amount'].mean() if len(as_receiver) > 0 else 0,  # avg receive amount
                as_sender['amount'].std() if len(as_sender) > 1 else 0,  # std send
                as_receiver['amount'].std() if len(as_receiver) > 1 else 0,  # std receive
                as_sender['amount'].sum() if len(as_sender) > 0 else 0,  # total sent
                as_receiver['amount'].sum() if len(as_receiver) > 0 else 0,  # total received
                len(as_sender) / (len(as_sender) + len(as_receiver) + 1),  # send ratio
            ]
            node_features.append(features)
        
        return np.array(node_features)
    
    def _create_edge_features(self):
        """Create edge-level features"""
        edge_features = []
        
        for idx, row in self.df.iterrows():
            features = [
                row['amount'],
                np.log(row['amount'] + 1),  # log amount
                row['hour'] if 'hour' in row else 0,
                row['day_of_week'] if 'day_of_week' in row else 0,
                1 if row['amount'] % 100 == 0 else 0,  # round number flag
            ]
            edge_features.append(features)
        
        return np.array(edge_features)
    
    def _create_node_labels(self):
        """Create node labels (1 if account involved in fraud)"""
        fraud_accounts = set()
        
        # Accounts that sent fraudulent transactions
        fraud_tx = self.df[self.df['is_fraud'] == 1]
        fraud_accounts.update(fraud_tx['from_account'].tolist())
        fraud_accounts.update(fraud_tx['to_account'].tolist())
        
        labels = []
        for account in self.account_encoder.classes_:
            labels.append(1 if account in fraud_accounts else 0)
        
        return np.array(labels)

if __name__ == "__main__":
    # Test with synthetic data
    from synthetic_data_generator import SyntheticUPIDataGenerator
    
    generator = SyntheticUPIDataGenerator(100, 1000)
    df = generator.generate_dataset()
    
    builder = TransactionGraphBuilder(df)
    graph = builder.build_graph()
    
    print(f"Graph built successfully!")
    print(f"Number of nodes: {graph['account'].x.shape[0]}")
    print(f"Number of edges: {graph['account', 'transacts', 'account'].edge_index.shape[1]}")
    print(f"Node features shape: {graph['account'].x.shape}")
    print(f"Edge features shape: {graph['account', 'transacts', 'account'].edge_attr.shape}")