"""
Pydantic schemas for API request/response
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Transaction(BaseModel):
    from_account: str
    to_account: str
    amount: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_account": "UPI_000001",
                "to_account": "UPI_000002",
                "amount": 5000.0,
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class BatchTransactionRequest(BaseModel):
    transactions: List[Transaction]
    batch_id: Optional[str] = None

class RiskScoreResponse(BaseModel):
    account_id: str
    risk_score: float
    risk_level: str  # Low, Medium, High, Critical
    confidence: float

class PredictionResponse(BaseModel):
    batch_id: str
    predictions: List[RiskScoreResponse]
    flagged_accounts: List[str]
    summary: Dict[str, int]

class ExplanationRequest(BaseModel):
    account_id: str
    top_k_features: Optional[int] = 5

class ExplanationResponse(BaseModel):
    account_id: str
    risk_score: float
    important_features: List[Dict[str, float]]
    pattern_detected: Optional[str]
    explanation_text: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str