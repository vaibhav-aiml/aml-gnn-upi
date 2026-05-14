"""
Test fixtures - Sample data for testing
"""

import json
from pathlib import Path

def load_sample_transactions():
    """Load sample transaction data for tests"""
    fixture_path = Path(__file__).parent / "sample_transactions.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)

__all__ = ['load_sample_transactions']