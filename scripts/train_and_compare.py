"""
Train and compare all GNN models
"""

from src.data.synthetic_data_generator import SyntheticUPIDataGenerator
from src.data.graph_builder import TransactionGraphBuilder
from src.models.model_comparison import ModelComparison
import torch

def main():
    print("=" * 60)
    print("MULTI-MODEL COMPARISON")
    print("GraphSAGE vs GAT vs Hetero-GNN")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n[1] Generating data...")
    generator = SyntheticUPIDataGenerator(num_accounts=150, num_transactions=3000)
    df = generator.generate_dataset()
    
    # Step 2: Build graph
    print("\n[2] Building graph...")
    builder = TransactionGraphBuilder(df)
    graph = builder.build_graph()
    
    # Prepare data
    data = graph
    data.x = graph['account'].x
    data.edge_index = graph['account', 'transacts', 'account'].edge_index
    data.y = graph['account'].y
    
    print(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Step 3: Train and compare
    print("\n[3] Training all models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    comparer = ModelComparison(data, device=device)
    results = comparer.train_all_models(epochs=30, lr=0.01)
    
    # Step 4: Show comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    df_results = comparer.get_comparison_dataframe()
    print(df_results.to_string(index=False))
    
    # Step 5: Save models
    print("\n[5] Saving models...")
    comparer.save_all_models()
    
    # Step 6: Show best model
    best_model, _ = comparer.get_best_model()
    print(f"\nBest Model: {best_model}")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)
    
    return comparer, df_results

if __name__ == "__main__":
    comparer, results = main()