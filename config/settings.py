"""
Configuration settings for the AML GNN project
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models_saved"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Neo4j configuration settings (RESERVED - Optional external graph database sync)
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")  # DEV-ONLY FALLBACK - DO NOT USE IN PRODUCTION

# Model hyperparameters
MODEL_CONFIG = {
    "graphsage": {
        "in_channels": 9,  # Matches 9 node features from TransactionGraphBuilder
        "hidden_channels": 256,
        "out_channels": 2,
        "num_layers": 3,
    },
    "gat": {
        "in_channels": 9,  # Matches 9 node features from TransactionGraphBuilder
        "hidden_channels": 256,
        "out_channels": 2,
        "heads": 4,
        "dropout": 0.6,
    },
    "training": {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 1024,
        "weight_decay": 5e-4,
    }
}

# Fraud patterns to detect
FRAUD_PATTERNS = {
    "smurfing": {
        "description": "Single source splitting to many accounts",
        "min_splits": 20,
        "min_amount": 1000,
    },
    "layering": {
        "description": "Money hopping through multiple accounts",
        "min_depth": 5,
    },
    "round_tripping": {
        "description": "Money returns to source after multiple hops",
        "min_hops": 3,
    }
}

# Settings class for easy access
class Settings:
    def __init__(self):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.data_path = DATA_DIR
        self.model_path = MODEL_DIR
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR

settings = Settings()