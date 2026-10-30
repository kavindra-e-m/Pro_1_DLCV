from fastapi.testclient import TestClient
from backend.main import app

def test_health_and_dataset_status():
    with TestClient(app) as client:
        health=client.get("/api/health")
        assert health.status_code==200
        assert health.json()["status"]=="ok"
        stats=client.get("/api/dataset/statistics")
        assert stats.status_code==200

def test_upload_rejects_non_image():
    with TestClient(app) as client:
        response=client.post("/api/search",files={"file":("notes.txt",b"plain text","text/plain")})
        assert response.status_code==415
