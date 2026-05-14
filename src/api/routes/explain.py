"""
Explanation endpoints
"""
from fastapi import APIRouter, HTTPException
from src.api.schemas import ExplanationRequest, ExplanationResponse

router = APIRouter()

@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Get explanation for a prediction"""
    
    # Mock explanation (replace with actual GNNExplainer)
    explanation_text = f"""
    Account {request.account_id} was flagged as suspicious due to:
    1. High transaction velocity (45 transactions in last hour)
    2. Round number amounts detected in 80% of transactions
    3. Connected to 3 known high-risk accounts
    4. Unusual transaction timing (active between 2-4 AM)
    """
    
    important_features = [
        {"transaction_velocity": 45.0},
        {"round_amount_ratio": 0.8},
        {"high_risk_connections": 3},
        {"unusual_timing_score": 0.9},
        {"amount_volatility": 0.7}
    ][:request.top_k_features]
    
    return ExplanationResponse(
        account_id=request.account_id,
        risk_score=85.5,
        important_features=important_features,
        pattern_detected="Smurfing pattern detected",
        explanation_text=explanation_text
    )

@router.get("/explain/features")
async def get_available_features():
    """Get list of available features for explanation"""
    return {
        "features": [
            "transaction_velocity",
            "round_amount_ratio",
            "high_risk_connections",
            "unusual_timing_score",
            "amount_volatility",
            "degree_centrality",
            "betweenness_centrality",
            "transaction_pattern_similarity"
        ]
    }