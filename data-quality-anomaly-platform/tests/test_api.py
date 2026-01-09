from fastapi.testclient import TestClient
import pandas as pd
import io
import sys
import os

# Ensure app can be imported
sys.path.append(os.getcwd())
from app.main import app

client = TestClient(app)

def get_auth_header():
    # Use form data for OAuth2
    response = client.post("/api/token", data={"username": "testuser", "password": "password"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Data Quality & Anomaly Platform"}

def test_drift_endpoint():
    # Create dummy CSVs
    df1 = pd.DataFrame({'val': [1.0, 1.1, 1.2, 1.3, 1.4]})
    df2 = pd.DataFrame({'val': [1.0, 1.1, 1.2, 1.3, 2.5]}) # Drifted
    
    csv1 = df1.to_csv(index=False).encode('utf-8')
    csv2 = df2.to_csv(index=False).encode('utf-8')
    
    files = {
        'reference_file': ('ref.csv', csv1, 'text/csv'),
        'current_file': ('curr.csv', csv2, 'text/csv')
    }
    
    headers = get_auth_header()
    response = client.post("/api/drift", files=files, headers=headers)
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "success"
    assert "drift_report" in json_resp
    
    # Check if drift details are present
    details = json_resp["drift_report"]["details"]
    assert "val" in details
