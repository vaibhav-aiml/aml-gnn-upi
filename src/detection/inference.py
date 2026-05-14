"""
Real-time inference for fraud detection
"""
import torch
import numpy as np
from collections import deque
import time

class RealTimeInference:
    def __init__(self, model, graph_builder, batch_size=100):
        self.model = model
        self.graph_builder = graph_builder
        self.batch_size = batch_size
        self.transaction_buffer = deque(maxlen=10000)
        self.risk_cache = {}
        
    def process_transaction(self, transaction):
        """Process single transaction in real-time"""
        # Add to buffer
        self.transaction_buffer.append(transaction)
        
        # Get account risk score
        from_risk = self.get_account_risk(transaction['from_account'])
        to_risk = self.get_account_risk(transaction['to_account'])
        
        # Calculate transaction risk
        tx_risk = self.calculate_transaction_risk(transaction, from_risk, to_risk)
        
        # Generate alert if high risk
        if tx_risk > 70:
            alert = self.generate_alert(transaction, tx_risk)
            return alert
        
        return None
    
    def get_account_risk(self, account_id):
        """Get cached risk score for account"""
        if account_id in self.risk_cache:
            # Check if cache is stale (> 5 minutes)
            if time.time() - self.risk_cache[account_id]['timestamp'] < 300:
                return self.risk_cache[account_id]['score']
        
        # Compute new risk score
        risk = self.compute_account_risk(account_id)
        self.risk_cache[account_id] = {
            'score': risk,
            'timestamp': time.time()
        }
        return risk
    
    def compute_account_risk(self, account_id):
        """Compute risk score for an account"""
        # Get recent transactions for this account
        account_tx = [tx for tx in self.transaction_buffer 
                     if tx['from_account'] == account_id or tx['to_account'] == account_id]
        
        if len(account_tx) < 5:
            return 0.0
        
        # Features for risk calculation
        features = {
            'velocity': len(account_tx) / 60,  # transactions per minute
            'avg_amount': np.mean([tx['amount'] for tx in account_tx]),
            'round_ratio': sum(1 for tx in account_tx if tx['amount'] % 1000 == 0) / len(account_tx),
            'unique_counterparties': len(set([tx['from_account'] for tx in account_tx] + 
                                             [tx['to_account'] for tx in account_tx]))
        }
        
        # Weighted risk score
        risk = (
            min(features['velocity'] / 10, 1.0) * 30 +  # Velocity: max 30 points
            features['round_ratio'] * 20 +               # Round numbers: max 20 points
            min(features['unique_counterparties'] / 20, 1.0) * 30  # Diversity: max 30 points
        )
        
        return min(risk * 100, 100)  # Scale to 0-100
    
    def calculate_transaction_risk(self, transaction, from_risk, to_risk):
        """Calculate risk score for a transaction"""
        # Base risk from accounts
        base_risk = (from_risk + to_risk) / 2
        
        # Transaction-specific factors
        amount_factor = min(transaction['amount'] / 100000, 1.0) * 20
        round_factor = 20 if transaction['amount'] % 1000 == 0 else 0
        
        total_risk = base_risk * 0.6 + amount_factor + round_factor
        
        return min(total_risk, 100)
    
    def generate_alert(self, transaction, risk_score):
        """Generate alert for suspicious transaction"""
        alert = {
            'timestamp': time.time(),
            'transaction': transaction,
            'risk_score': risk_score,
            'alert_level': 'HIGH' if risk_score > 85 else 'MEDIUM',
            'reason': self.get_risk_reasons(transaction, risk_score)
        }
        return alert
    
    def get_risk_reasons(self, transaction, risk_score):
        """Get human-readable reasons for risk score"""
        reasons = []
        
        if transaction['amount'] % 1000 == 0:
            reasons.append("Round number amount detected")
        
        if transaction['amount'] > 50000:
            reasons.append("Large transaction amount")
        
        from_risk = self.get_account_risk(transaction['from_account'])
        to_risk = self.get_account_risk(transaction['to_account'])
        
        if from_risk > 70:
            reasons.append(f"Source account has high risk ({from_risk:.0f})")
        
        if to_risk > 70:
            reasons.append(f"Destination account has high risk ({to_risk:.0f})")
        
        return reasons
    
    def batch_process(self, transactions):
        """Process batch of transactions"""
        alerts = []
        for tx in transactions:
            alert = self.process_transaction(tx)
            if alert:
                alerts.append(alert)
        return alerts

class BatchInference:
    """Batch inference for periodic processing"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_batch(self, graph_data):
        """Run inference on batch of graphs"""
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            logits = self.model(graph_data.x, graph_data.edge_index)
            probabilities = torch.softmax(logits, dim=1)
            risk_scores = probabilities[:, 1] * 100
        return risk_scores.cpu().numpy()

if __name__ == "__main__":
    print("Real-time inference module ready")