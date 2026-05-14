"""
Data preprocessing utilities for transaction data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta

class TransactionPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.account_encoder = LabelEncoder()
        
    def clean_data(self, df):
        """Clean raw transaction data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna(subset=['from_account', 'to_account', 'amount'])
        
        # Remove self-transactions
        df = df[df['from_account'] != df['to_account']]
        
        # Remove negative or zero amounts
        df = df[df['amount'] > 0]
        
        # Cap extreme outliers (99.9th percentile)
        amount_cap = df['amount'].quantile(0.999)
        df['amount'] = df['amount'].clip(upper=amount_cap)
        
        return df
    
    def add_temporal_features(self, df):
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        else:
            # Add dummy features if timestamp missing
            df['hour'] = 12
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['month'] = 1
            df['is_weekend'] = 0
            df['is_night'] = 0
        
        return df
    
    def add_amount_features(self, df):
        """Add amount-based features"""
        df['log_amount'] = np.log1p(df['amount'])
        df['sqrt_amount'] = np.sqrt(df['amount'])
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
        df['is_small_amount'] = (df['amount'] < 1000).astype(int)
        df['is_large_amount'] = (df['amount'] > 50000).astype(int)
        df['amount_bucket'] = pd.cut(df['amount'], 
                                      bins=[0, 1000, 5000, 10000, 50000, float('inf')],
                                      labels=[1, 2, 3, 4, 5])
        
        return df
    
    def add_aggregate_features(self, df):
        """Add account-level aggregate features"""
        # Per account statistics
        for account_col in ['from_account', 'to_account']:
            # Transaction count per account
            tx_count = df.groupby(account_col).size().to_dict()
            df[f'{account_col}_tx_count'] = df[account_col].map(tx_count)
            
            # Average amount per account
            avg_amount = df.groupby(account_col)['amount'].mean().to_dict()
            df[f'{account_col}_avg_amount'] = df[account_col].map(avg_amount)
            
            # Total amount per account
            total_amount = df.groupby(account_col)['amount'].sum().to_dict()
            df[f'{account_col}_total_amount'] = df[account_col].map(total_amount)
        
        # Ratio features
        df['to_from_ratio'] = df['to_account_tx_count'] / (df['from_account_tx_count'] + 1)
        df['avg_amount_ratio'] = df['to_account_avg_amount'] / (df['from_account_avg_amount'] + 1)
        
        return df
    
    def normalize_features(self, df, feature_columns):
        """Normalize numerical features"""
        available_features = [col for col in feature_columns if col in df.columns]
        df[available_features] = self.scaler.fit_transform(df[available_features])
        return df
    
    def handle_class_imbalance(self, df, target_col='is_fraud', method='smote'):
        """Handle class imbalance for fraud detection"""
        fraud_count = df[df[target_col] == 1].shape[0]
        legit_count = df[df[target_col] == 0].shape[0]
        
        print(f"Original class distribution:")
        print(f"  Legit: {legit_count} ({legit_count/(legit_count+fraud_count)*100:.2f}%)")
        print(f"  Fraud: {fraud_count} ({fraud_count/(legit_count+fraud_count)*100:.2f}%)")
        
        if method == 'undersample':
            # Undersample majority class
            legit_sample = df[df[target_col] == 0].sample(n=fraud_count, random_state=42)
            fraud_sample = df[df[target_col] == 1]
            df_balanced = pd.concat([legit_sample, fraud_sample])
            
        elif method == 'oversample':
            # Oversample minority class
            legit_sample = df[df[target_col] == 0]
            fraud_sample = df[df[target_col] == 1].sample(n=legit_count, replace=True, random_state=42)
            df_balanced = pd.concat([legit_sample, fraud_sample])
            
        else:  # smote - simplified version
            legit_sample = df[df[target_col] == 0].sample(n=fraud_count * 3, random_state=42)
            fraud_sample = df[df[target_col] == 1]
            df_balanced = pd.concat([legit_sample, fraud_sample])
        
        print(f"Balanced class distribution:")
        df_balanced[target_col].value_counts().apply(lambda x: print(f"  {x}"))
        
        return df_balanced
    
    def create_sequences(self, df, account_col='from_account', sequence_length=10):
        """Create transaction sequences per account for LSTM models"""
        sequences = []
        labels = []
        
        for account in df[account_col].unique():
            account_tx = df[df[account_col] == account].sort_values('timestamp')
            
            if len(account_tx) >= sequence_length:
                for i in range(len(account_tx) - sequence_length + 1):
                    seq = account_tx.iloc[i:i+sequence_length][['amount', 'hour', 'day_of_week']].values
                    sequences.append(seq)
                    labels.append(account_tx.iloc[i+sequence_length-1]['is_fraud'])
        
        return np.array(sequences), np.array(labels)
    
    def preprocess_pipeline(self, df):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Step 1: Clean data
        df = self.clean_data(df)
        print(f"After cleaning: {len(df)} transactions")
        
        # Step 2: Add temporal features
        df = self.add_temporal_features(df)
        
        # Step 3: Add amount features
        df = self.add_amount_features(df)
        
        # Step 4: Add aggregate features
        df = self.add_aggregate_features(df)
        
        print(f"Final feature count: {len(df.columns)}")
        
        return df
    
    def split_data(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split data into train/val/test sets"""
        np.random.seed(42)
        
        indices = np.random.permutation(len(df))
        train_end = int(train_ratio * len(df))
        val_end = int((train_ratio + val_ratio) * len(df))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
        
        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

# Additional utility functions
def preprocess_transactions(df):
    """Quick preprocessing function"""
    preprocessor = TransactionPreprocessor()
    return preprocessor.preprocess_pipeline(df)

def normalize_features(df, feature_columns):
    """Quick normalization function"""
    preprocessor = TransactionPreprocessor()
    return preprocessor.normalize_features(df, feature_columns)

def validate_transaction(transaction):
    """Validate a single transaction"""
    required_fields = ['from_account', 'to_account', 'amount']
    
    for field in required_fields:
        if field not in transaction:
            return False, f"Missing field: {field}"
    
    if transaction['amount'] <= 0:
        return False, "Amount must be positive"
    
    if transaction['from_account'] == transaction['to_account']:
        return False, "Self-transactions not allowed"
    
    return True, "Valid"

if __name__ == "__main__":
    # Test with sample data
    from synthetic_data_generator import SyntheticUPIDataGenerator
    
    generator = SyntheticUPIDataGenerator(100, 1000)
    df = generator.generate_dataset()
    
    preprocessor = TransactionPreprocessor()
    processed_df = preprocessor.preprocess_pipeline(df)
    
    print(f"\nPreprocessed data shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)[:10]}...")