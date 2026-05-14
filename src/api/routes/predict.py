"""
Prediction endpoints
"""
from fastapi import APIRouter, HTTPException
from src.api.schemas import BatchTransactionRequest, PredictionResponse, RiskScoreResponse
import torch
import numpy as np
from typing import List

router = APIRouter()

# Mock model for demonstration (replace with actual model)
class MockModel:
    def predict(self, account):
        import random
        return random.uniform(0, 100)

model = MockModel()

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: BatchTransactionRequest):
    """Predict fraud risk for batch of transactions"""
    try:
        # Extract unique accounts
        accounts = set()
        for tx in request.transactions:
            accounts.add(tx.from_account)
            accounts.add(tx.to_account)
        
        # Generate predictions
        predictions = []
        flagged = []
        
        for account in accounts:
            risk = model.predict(account)
            risk_score = float(risk)
            
            # Determine risk level
            if risk_score < 30:
                level = "Low"
            elif risk_score < 60:
                level = "Medium"
            elif risk_score < 85:
                level = "High"
            else:
                level = "Critical"
            
            predictions.append(RiskScoreResponse(
                account_id=account,
                risk_score=risk_score,
                risk_level=level,
                confidence=0.85
            ))
            
            if risk_score > 70:
                flagged.append(account)
        
        summary = {
            "total_accounts": len(accounts),
            "flagged_count": len(flagged),
            "avg_risk": np.mean([p.risk_score for p in predictions])
        }
        
        return PredictionResponse(
            batch_id=request.batch_id or "batch_001",
            predictions=predictions,
            flagged_accounts=flagged,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/account/{account_id}")
async def predict_single_account(account_id: str):
    """Predict risk for single account"""
    risk = model.predict(account_id)
    return {
        "account_id": account_id,
        "risk_score": risk,
        "risk_level": "High" if risk > 70 else "Low"
    }