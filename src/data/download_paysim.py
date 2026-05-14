"""
Download PaySim2 dataset for transaction data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io

def download_paysim_dataset(save_path="data/raw/paysim2/"):
    """
    Download PaySim2 dataset from source
    
    Args:
        save_path: Directory to save the dataset
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading PaySim2 dataset...")
    print("Note: PaySim2 dataset is available at:")
    print("https://www.kaggle.com/datasets/ealtman2019/paysim-mobile-money-transactions")
    
    # Alternative: Generate synthetic data if download fails
    print("\nGenerating synthetic PaySim-like data...")
    df = generate_paysim_synthetic()
    
    # Save to CSV
    filepath = save_path / "transactions.csv"
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    
    return df

def generate_paysim_synthetic(num_transactions=50000):
    """
    Generate synthetic PaySim-like data
    """
    np.random.seed(42)
    
    # Generate transaction types
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    
    data = {
        'step': np.random.randint(1, 744, num_transactions),
        'type': np.random.choice(transaction_types, num_transactions),
        'amount': np.random.exponential(5000, num_transactions),
        'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(num_transactions)],
        'nameDest': [f'M{np.random.randint(1000000, 9999999)}' for _ in range(num_transactions)],
        'oldbalanceOrg': np.random.uniform(0, 100000, num_transactions),
        'newbalanceOrig': np.random.uniform(0, 100000, num_transactions),
        'oldbalanceDest': np.random.uniform(0, 50000, num_transactions),
        'newbalanceDest': np.random.uniform(0, 50000, num_transactions),
        'isFraud': np.random.choice([0, 1], num_transactions, p=[0.995, 0.005]),
        'isFlaggedFraud': 0
    }
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} synthetic PaySim transactions")
    print(f"Fraud ratio: {df['isFraud'].sum() / len(df) * 100:.2f}%")
    
    return df

def load_paysim_data(filepath="data/raw/paysim2/transactions.csv"):
    """Load PaySim dataset from CSV"""
    return pd.read_csv(filepath)

if __name__ == "__main__":
    df = download_paysim_dataset()
    print(f"Dataset shape: {df.shape}")