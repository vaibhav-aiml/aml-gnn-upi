"""
Generate synthetic UPI transaction data with fraud patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SyntheticUPIDataGenerator:
    def __init__(self, num_accounts=1000, num_transactions=50000):
        self.num_accounts = num_accounts
        self.num_transactions = num_transactions
        self.accounts = [f"UPI_{i:06d}" for i in range(num_accounts)]
        
    def generate_normal_transactions(self, num_normal):
        """Generate legitimate transactions"""
        transactions = []
        for _ in range(num_normal):
            from_acc = random.choice(self.accounts)
            to_acc = random.choice([a for a in self.accounts if a != from_acc])
            
            transactions.append({
                'from_account': from_acc,
                'to_account': to_acc,
                'amount': random.uniform(10, 5000),
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 30)),
                'is_fraud': 0,
                'pattern_type': 'normal'
            })
        return transactions
    
    def generate_smurfing_pattern(self, num_patterns=5):
        """Generate smurfing: one source to many small accounts"""
        transactions = []
        for _ in range(num_patterns):
            source = random.choice(self.accounts)
            num_splits = random.randint(20, 50)
            total_amount = random.uniform(50000, 200000)
            split_amount = total_amount / num_splits
            
            recipients = random.sample([a for a in self.accounts if a != source], num_splits)
            for recipient in recipients:
                transactions.append({
                    'from_account': source,
                    'to_account': recipient,
                    'amount': split_amount,
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 7)),
                    'is_fraud': 1,
                    'pattern_type': 'smurfing'
                })
        return transactions
    
    def generate_layering_pattern(self, num_patterns=5):
        """Generate layering: money through chain of accounts"""
        transactions = []
        for _ in range(num_patterns):
            chain_length = random.randint(5, 10)
            chain = random.sample(self.accounts, chain_length)
            amount = random.uniform(10000, 100000)
            
            for i in range(len(chain)-1):
                transactions.append({
                    'from_account': chain[i],
                    'to_account': chain[i+1],
                    'amount': amount,
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 7)),
                    'is_fraud': 1,
                    'pattern_type': 'layering'
                })
        return transactions
    
    def generate_round_tripping(self, num_patterns=5):
        """Generate round-tripping: money returns to source"""
        transactions = []
        for _ in range(num_patterns):
            source = random.choice(self.accounts)
            intermediaries = random.sample([a for a in self.accounts if a != source], random.randint(3, 7))
            amount = random.uniform(5000, 50000)
            
            # Source to first intermediary
            transactions.append({
                'from_account': source,
                'to_account': intermediaries[0],
                'amount': amount,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 7)),
                'is_fraud': 1,
                'pattern_type': 'round_tripping'
            })
            
            # Through intermediaries
            for i in range(len(intermediaries)-1):
                transactions.append({
                    'from_account': intermediaries[i],
                    'to_account': intermediaries[i+1],
                    'amount': amount,
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 7)),
                    'is_fraud': 1,
                    'pattern_type': 'round_tripping'
                })
            
            # Back to source
            transactions.append({
                'from_account': intermediaries[-1],
                'to_account': source,
                'amount': amount,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 7)),
                'is_fraud': 1,
                'pattern_type': 'round_tripping'
            })
        return transactions
    
    def generate_dataset(self):
        """Generate complete dataset with fraud patterns"""
        print("Generating synthetic UPI transaction data...")
        
        # Generate transactions
        num_fraud = int(self.num_transactions * 0.05)  # 5% fraud
        num_normal = self.num_transactions - num_fraud
        
        transactions = self.generate_normal_transactions(num_normal)
        transactions.extend(self.generate_smurfing_pattern(3))
        transactions.extend(self.generate_layering_pattern(3))
        transactions.extend(self.generate_round_tripping(3))
        
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        # Add features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        print(f"Generated {len(df)} transactions")
        print(f"Fraud ratio: {df['is_fraud'].sum() / len(df) * 100:.2f}%")
        
        return df

if __name__ == "__main__":
    generator = SyntheticUPIDataGenerator(num_accounts=500, num_transactions=10000)
    df = generator.generate_dataset()
    df.to_csv("data/raw/synthetic_upi_transactions.csv", index=False)
    print("Dataset saved to data/raw/synthetic_upi_transactions.csv")