"""
Custom loss functions for imbalanced fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses on hard-to-classify examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class WeightedBinaryCrossEntropy(nn.Module):
    """Weighted BCE for imbalanced datasets"""
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        
    def forward(self, inputs, targets):
        # Convert to binary format
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
        
        loss = F.binary_cross_entropy_with_logits(
            inputs[:, 1], targets.float(), pos_weight=self.pos_weight.to(inputs.device)
        )
        return loss

class CombinedLoss(nn.Module):
    """Combine multiple loss functions"""
    def __init__(self, alpha=0.5, focal_gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.weighted_bce = WeightedBinaryCrossEntropy(pos_weight=pos_weight)
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        return self.alpha * self.focal(inputs, targets) + \
               (1 - self.alpha) * self.weighted_bce(inputs, targets)

if __name__ == "__main__":
    # Test losses
    batch_size = 32
    num_classes = 2
    
    predictions = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, 2, (batch_size,))
    
    focal = FocalLoss()
    wbce = WeightedBinaryCrossEntropy()
    combined = CombinedLoss()
    
    print(f"Focal Loss: {focal(predictions, labels):.4f}")
    print(f"Weighted BCE: {wbce(predictions, labels):.4f}")
    print(f"Combined Loss: {combined(predictions, labels):.4f}")