"""
Complete pipeline runner for AML GNN project
"""

from src.data.synthetic_data_generator import SyntheticUPIDataGenerator
from src.data.graph_builder import TransactionGraphBuilder
from src.models.graphsage import GraphSAGELight
from src.models.trainer import GNNTrainer
from src.detection.patterns import PatternDetector
import torch
import networkx as nx

import argparse

def main():
    parser = argparse.ArgumentParser(description="AML GNN Pipeline Runner")
    parser.add_argument("--fast", action="store_true", help="Run fast pipeline demo with reduced graph size")
    args = parser.parse_args()

    num_accounts = 200 if args.fast else 500
    num_tx = 5000 if args.fast else 10000

    print("=" * 60)
    print(f"AML GNN Project Pipeline {'(Fast Mode)' if args.fast else ''}")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic transaction data...")
    generator = SyntheticUPIDataGenerator(num_accounts=num_accounts, num_transactions=num_tx)
    df = generator.generate_dataset()
    print(f"Generated {len(df)} transactions")
    
    # Step 2: Build graph
    print("\n[Step 2] Building graph structure...")
    builder = TransactionGraphBuilder(df)
    graph = builder.build_graph()
    print(f"Graph built: {graph['account'].x.shape[0]} nodes, "
          f"{graph['account', 'transacts', 'account'].edge_index.shape[1]} edges")
    
    # Step 3: Create NetworkX graph for pattern detection
    print("\n[Step 3] Creating NetworkX graph for pattern detection...")
    nx_graph = nx.DiGraph()
    for _, row in df.iterrows():
        nx_graph.add_edge(
            row['from_account'], 
            row['to_account'], 
            amount=row['amount'],
            timestamp=row['timestamp']
        )
    
    # Step 4: Detect patterns
    print("\n[Step 4] Detecting fraud patterns...")
    detector = PatternDetector(nx_graph)
    patterns = detector.get_all_patterns()
    
    # Step 5: Train GNN model
    print("\n[Step 5] Training GraphSAGE model...")
    model = GraphSAGELight(
        in_channels=graph['account'].x.shape[1],
        hidden_channels=64,
        out_channels=2
    )
    
    trainer = GNNTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {trainer.device}")
    
    # Prepare data
    data = graph
    data.x = graph['account'].x
    data.edge_index = graph['account', 'transacts', 'account'].edge_index
    data.y = graph['account'].y
    
    # Train (just a few epochs for demo)
    print("\nTraining model (10 epochs for demo)...")
    trainer.train(data, epochs=10, lr=0.01)
    
    # Step 6: Save model
    print("\n[Step 6] Saving model...")
    trainer.save_model("models_saved/graphsage_model.pt")
    
    # Step 7: Generate risk scores
    print("\n[Step 7] Generating risk scores...")
    risk_scores = trainer.model.predict_risk_score(data.x, data.edge_index)
    high_risk_nodes = torch.where(risk_scores > 70)[0]
    print(f"High-risk accounts detected: {len(high_risk_nodes)}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return {
        'transactions': df,
        'graph': graph,
        'patterns': patterns,
        'model': model,
        'risk_scores': risk_scores
    }

if __name__ == "__main__":
    results = main()