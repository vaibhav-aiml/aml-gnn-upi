"""
Benchmark model scalability on different graph sizes
"""
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_data_generator import SyntheticUPIDataGenerator
from src.data.graph_builder import TransactionGraphBuilder
from src.models.graphsage import GraphSAGELight

class ScalabilityBenchmark:
    def __init__(self):
        self.results = []
        
    def benchmark_graph_size(self, num_accounts_list=[100, 500, 1000, 2000]):
        """Benchmark performance on different graph sizes"""
        
        for num_accounts in num_accounts_list:
            print(f"\nBenchmarking with {num_accounts} accounts...")
            
            # Generate data
            generator = SyntheticUPIDataGenerator(
                num_accounts=num_accounts,
                num_transactions=num_accounts * 10
            )
            df = generator.generate_dataset()
            
            # Build graph
            start_time = time.time()
            builder = TransactionGraphBuilder(df)
            graph = builder.build_graph()
            graph_build_time = time.time() - start_time
            
            # Model inference
            model = GraphSAGELight(
                in_channels=graph['account'].x.shape[1],
                hidden_channels=64,
                out_channels=2
            )
            
            start_time = time.time()
            with torch.no_grad():
                output = model(graph['account'].x, graph['account', 'transacts', 'account'].edge_index)
            inference_time = time.time() - start_time
            
            # Memory usage
            memory_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.results.append({
                'num_accounts': num_accounts,
                'num_transactions': len(df),
                'graph_build_time': graph_build_time,
                'inference_time': inference_time,
                'memory_mb': memory_mb
            })
            
        return pd.DataFrame(self.results)
    
    def print_summary(self, df):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        
        print("\n" + "="*60)
        print("SCALABILITY ANALYSIS")
        print("="*60)
        print(f"Time scaling factor: {df['inference_time'].iloc[-1] / df['inference_time'].iloc[0]:.2f}x")
        print(f"Graph size growth: {df['num_accounts'].iloc[-1] / df['num_accounts'].iloc[0]:.0f}x")
        
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting scalability benchmark...")
        results_df = self.benchmark_graph_size()
        self.print_summary(results_df)
        
        # Save results
        results_df.to_csv("models_saved/benchmark_results.csv", index=False)
        print("\n✅ Results saved to models_saved/benchmark_results.csv")
        
        return results_df

if __name__ == "__main__":
    benchmark = ScalabilityBenchmark()
    results = benchmark.run_full_benchmark()