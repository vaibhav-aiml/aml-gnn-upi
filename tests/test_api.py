"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class TestAPI:
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        
    def test_health_check(self):
        """Test health endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        test_data = {
            "transactions": [
                {
                    "from_account": "UPI_000001",
                    "to_account": "UPI_000002",
                    "amount": 5000.0,
                    "timestamp": "2024-01-15T10:30:00"
                },
                {
                    "from_account": "UPI_000002",
                    "to_account": "UPI_000003",
                    "amount": 10000.0,
                    "timestamp": "2024-01-15T10:35:00"
                }
            ],
            "batch_id": "test_batch_001"
        }
        
        response = client.post("/api/v1/predict", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "flagged_accounts" in data
        assert "summary" in data
        
    def test_explain_endpoint(self):
        """Test explanation endpoint"""
        test_data = {
            "account_id": "UPI_000001",
            "top_k_features": 3
        }
        
        response = client.post("/api/v1/explain", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["account_id"] == "UPI_000001"
        assert "important_features" in data
        assert len(data["important_features"]) == 3
        
    def test_live_endpoint(self):
        """Test liveness probe"""
        response = client.get("/api/v1/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
        
    def test_ready_endpoint(self):
        """Test readiness probe"""
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])