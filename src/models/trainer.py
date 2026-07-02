"""
Training pipeline for GNN models
"""
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm

class GNNTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_metrics = []
        
    def train_epoch(self, data, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        """Evaluate the model"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # Calculate metrics
            accuracy = accuracy_score(data.y.cpu(), pred.cpu())
            precision = precision_score(data.y.cpu(), pred.cpu(), average='binary', zero_division=0)
            recall = recall_score(data.y.cpu(), pred.cpu(), average='binary', zero_division=0)
            f1 = f1_score(data.y.cpu(), pred.cpu(), average='binary', zero_division=0)
            
            # AUC-ROC
            probs = torch.softmax(out, dim=1)[:, 1].cpu()
            auc = roc_auc_score(data.y.cpu(), probs)
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }
    
    def train(self, data, epochs=100, lr=0.001, weight_decay=5e-4):
        """Full training loop"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"Training on {self.device}")
        print(f"Total epochs: {epochs}")
        print("-" * 50)
        
        for epoch in tqdm(range(epochs), desc="Training"):
            loss = self.train_epoch(data, optimizer, criterion)
            self.train_losses.append(loss)
            
            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                metrics = self.evaluate(data)
                self.val_metrics.append(metrics)
                
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Loss: {loss:.4f}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-Score: {metrics['f1_score']:.4f}")
                print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
                print("-" * 50)
        
        return self.train_losses, self.val_metrics
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        print(f"Model loaded from {path}")