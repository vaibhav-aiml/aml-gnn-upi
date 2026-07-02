"""
Real-time transaction generator
"""
import random
import time
import threading
from datetime import datetime, timedelta
import json

class TransactionGenerator:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.accounts = [f"UPI_{i:06d}" for i in range(1000)]
        self.running = False
        self.transaction_count = 0
        self.fraud_count = 0
        
    def generate_normal_transaction(self):
        """Generate a normal transaction"""
        from_acc = random.choice(self.accounts)
        to_acc = random.choice([a for a in self.accounts if a != from_acc])
        
        return {
            'from_account': from_acc,
            'to_account': to_acc,
            'amount': round(random.uniform(10, 5000), 2),
            'transaction_type': random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT']),
            'is_fraud': 0,
            'pattern_type': 'normal'
        }
    
    def generate_fraud_smurfing(self):
        """Generate smurfing transaction"""
        source = random.choice(self.accounts)
        recipients = random.sample([a for a in self.accounts if a != source], 3)
        
        transactions = []
        for recipient in recipients:
            transactions.append({
                'from_account': source,
                'to_account': recipient,
                'amount': round(random.uniform(8000, 12000), 2),
                'transaction_type': 'TRANSFER',
                'is_fraud': 1,
                'pattern_type': 'smurfing'
            })
        return transactions
    
    def generate_fraud_layering(self):
        """Generate layering transaction"""
        chain = random.sample(self.accounts, 5)
        amount = round(random.uniform(10000, 50000), 2)
        
        transactions = []
        for i in range(len(chain)-1):
            transactions.append({
                'from_account': chain[i],
                'to_account': chain[i+1],
                'amount': amount,
                'transaction_type': 'TRANSFER',
                'is_fraud': 1,
                'pattern_type': 'layering'
            })
        return transactions
    
    def generate_batch(self, batch_size=10):
        """Generate a batch of transactions"""
        transactions = []
        
        for _ in range(batch_size):
            # 5% fraud rate
            if random.random() < 0.05:
                if random.random() < 0.5:
                    fraud_tx = self.generate_fraud_smurfing()
                    transactions.extend(fraud_tx)
                    self.fraud_count += len(fraud_tx)
                else:
                    fraud_tx = self.generate_fraud_layering()
                    transactions.extend(fraud_tx)
                    self.fraud_count += len(fraud_tx)
            else:
                transactions.append(self.generate_normal_transaction())
                self.transaction_count += 1
        
        return transactions
    
    def run(self, interval=0.5, batch_size=5):
        """
        Continuously generate and publish transactions
        Args:
            interval: Time between batches (seconds)
            batch_size: Number of transactions per batch
        """
        self.running = True
        
        while self.running:
            try:
                batch = self.generate_batch(batch_size)
                for tx in batch:
                    self.redis_client.publish_transaction(tx)
                    
                    # Publish alert for fraud transactions
                    if tx.get('is_fraud', 0) == 1:
                        alert = {
                            'transaction': tx,
                            'risk_score': random.randint(70, 100),
                            'alert_level': 'HIGH' if random.random() > 0.5 else 'CRITICAL'
                        }
                        self.redis_client.publish_alert(alert)
                
                if self.transaction_count % 100 == 0 and self.transaction_count > 0:
                    print(f"📊 Generated {self.transaction_count} transactions "
                          f"({self.fraud_count} fraud)")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Generation error: {e}")
                time.sleep(1)
    
    def stop(self):
        self.running = False