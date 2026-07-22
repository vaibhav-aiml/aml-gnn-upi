"""
Training pipeline for GNN models with stratified node splits
"""
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def add_stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create stratified train/val/test masks on node labels y.
    Ensures positive fraud labels are represented across all splits.
    """
    num_nodes = data.y.shape[0]
    y_numpy = data.y.cpu().numpy()
    
    indices = np.arange(num_nodes)
    
    unique_classes, counts = np.unique(y_numpy, return_counts=True)
    can_stratify = len(unique_classes) > 1 and min(counts) >= 3
    
    stratify_y = y_numpy if can_stratify else None
    
    train_idx, test_val_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=stratify_y,
        random_state=random_state
    )
    
    val_share_of_test_val = val_ratio / (val_ratio + test_ratio)
    stratify_test_val = y_numpy[test_val_idx] if can_stratify else None
    
    val_idx, test_idx = train_test_split(
        test_val_idx,
        test_size=(1.0 - val_share_of_test_val),
        stratify=stratify_test_val,
        random_state=random_state
    )
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Class balance sanity checks
    print("[INFO] Stratified Node Split Summary:")
    for split_name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
        split_y = data.y[mask].cpu().numpy()
        pos_count = (split_y == 1).sum()
        total = len(split_y)
        print(f"   {split_name} mask: {total} nodes ({pos_count} positive / {pos_count/total*100:.1f}%)")
        
    return data

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
        
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = criterion(out, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, mask=None):
        """Evaluate the model on target mask (defaults to test_mask/val_mask or full data)"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            if mask is None:
                if hasattr(data, 'test_mask') and data.test_mask is not None:
                    mask = data.test_mask
                elif hasattr(data, 'val_mask') and data.val_mask is not None:
                    mask = data.val_mask

            if mask is not None:
                y_true = data.y[mask].cpu()
                y_pred = pred[mask].cpu()
                logits_out = out[mask]
            else:
                y_true = data.y.cpu()
                y_pred = pred.cpu()
                logits_out = out

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # AUC-ROC
            probs = torch.softmax(logits_out, dim=1)[:, 1].cpu()
            try:
                auc = roc_auc_score(y_true, probs)
            except ValueError:
                auc = 0.5  # Fallback if single class present
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }
    
    def train(self, data, epochs=100, lr=0.001, weight_decay=5e-4):
        """Full training loop"""
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data = add_stratified_split(data)
            
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
                metrics = self.evaluate(data, mask=data.val_mask)
                self.val_metrics.append(metrics)
                
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Loss: {loss:.4f}")
                print(f"Val Accuracy: {metrics['accuracy']:.4f}")
                print(f"Val Precision: {metrics['precision']:.4f}")
                print(f"Val Recall: {metrics['recall']:.4f}")
                print(f"Val F1-Score: {metrics['f1_score']:.4f}")
                print(f"Val AUC-ROC: {metrics['auc_roc']:.4f}")
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
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        print(f"Model loaded from {path}")