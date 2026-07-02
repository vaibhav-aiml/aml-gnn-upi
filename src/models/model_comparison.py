"""
Multi-model comparison for AML fraud detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import time
from pathlib import Path

from .graphsage import GraphSAGELight, GraphSAGEFraudDetector
from .gat import GATFraudDetector
from .heterognn import SimpleHeteroGNN
from .trainer import GNNTrainer

class ModelComparison:
    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device
        self.results = {}
        self.models = {}
        self.train_times = {}
        
    def train_all_models(self, epochs=30, lr=0.01):
        """Train all three models and compare"""
        
        models_config = {
            'GraphSAGE': {
                'class': GraphSAGELight,
                'params': {
                    'in_channels': self.data.x.shape[1],
                    'hidden_channels': 64,
                    'out_channels': 2
                }
            },
            'GAT': {
                'class': GATFraudDetector,
                'params': {
                    'in_channels': self.data.x.shape[1],
                    'hidden_channels': 32,
                    'out_channels': 2,
                    'heads': 2,
                    'dropout': 0.3
                }
            },
            'Hetero-GNN': {
                'class': SimpleHeteroGNN,
                'params': {
                    'in_channels': self.data.x.shape[1],
                    'hidden_channels': 64,
                    'out_channels': 2
                }
            }
        }
        
        print("=" * 60)
        print("🚀 TRAINING ALL MODELS FOR COMPARISON")
        print("=" * 60)
        
        for model_name, config in models_config.items():
            print(f"\n📊 Training {model_name}...")
            
            # Initialize model
            model = config['class'](**config['params'])
            model = model.to(self.device)
            
            # Train
            start_time = time.time()
            trainer = GNNTrainer(model, self.device)
            losses, metrics = trainer.train(
                self.data, 
                epochs=epochs, 
                lr=lr
            )
            train_time = time.time() - start_time
            
            # Evaluate
            eval_metrics = trainer.evaluate(self.data)
            
            # Store results
            self.models[model_name] = {
                'model': model,
                'trainer': trainer,
                'losses': losses,
                'train_time': train_time
            }
            
            self.results[model_name] = {
                **eval_metrics,
                'train_time': train_time,
                'epochs': epochs
            }
            
            print(f"✅ {model_name} trained in {train_time:.2f}s")
            print(f"   Accuracy: {eval_metrics['accuracy']:.4f}")
            print(f"   F1-Score: {eval_metrics['f1_score']:.4f}")
        
        print("\n" + "=" * 60)
        print("✅ All models trained!")
        print("=" * 60)
        
        return self.results
    
    def get_comparison_dataframe(self):
        """Get results as pandas DataFrame"""
        df = pd.DataFrame(self.results).T
        df = df.reset_index().rename(columns={'index': 'Model'})
        return df
    
    def get_best_model(self, metric='f1_score'):
        """Get the best performing model"""
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        return best_model_name, self.models[best_model_name]['model']
    
    def save_all_models(self, save_dir="models_saved/comparison/"):
        """Save all trained models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_data in self.models.items():
            path = save_dir / f"{model_name}_model.pt"
            model_data['trainer'].save_model(str(path))
            print(f"✅ Saved {model_name} to {path}")
    
    def load_all_models(self, load_dir="models_saved/comparison/"):
        """Load all trained models"""
        load_dir = Path(load_dir)
        
        for model_name in self.models.keys():
            path = load_dir / f"{model_name}_model.pt"
            if path.exists():
                self.models[model_name]['trainer'].load_model(str(path))
                print(f"✅ Loaded {model_name} from {path}")