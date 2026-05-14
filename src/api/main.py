"""
FastAPI main application for AML service
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import predict, explain, health
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="AML Fraud Detection API",
    description="GNN-based anti-money laundering detection system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["predictions"])
app.include_router(explain.router, prefix="/api/v1", tags=["explanations"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.get("/")
async def root():
    return {
        "message": "AML Fraud Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/predict",
            "/api/v1/explain",
            "/api/v1/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)