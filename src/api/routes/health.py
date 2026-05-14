"""
Health check endpoints
"""
from fastapi import APIRouter
from src.api.schemas import HealthResponse
import time

router = APIRouter()
start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        uptime_seconds=uptime,
        version="1.0.0"
    )

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    return {"status": "ready"}

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}