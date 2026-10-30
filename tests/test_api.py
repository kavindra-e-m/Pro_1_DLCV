from fastapi.testclient import TestClient
from backend.main import app
from PIL import Image
import io

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

def test_upload_rejects_invalid_jpeg_bytes():
    with TestClient(app) as client:
        response=client.post("/api/search",files={"file":("bad.jpg",b"not an image","image/jpeg")})
        assert response.status_code==400

def test_upload_rejects_too_small_image():
    buffer=io.BytesIO();Image.new("RGB",(12,12)).save(buffer,format="JPEG")
    with TestClient(app) as client:
        response=client.post("/api/search",files={"file":("small.jpg",buffer.getvalue(),"image/jpeg")})
        assert response.status_code==400

def test_missing_checkpoint_is_reported_as_unavailable(tmp_path,monkeypatch):
    from backend.services.inference import InferenceService
    monkeypatch.setenv("BAGVISION_CHECKPOINT",str(tmp_path/"missing.pt"))
    instance=InferenceService()
    assert not instance.ready

def test_missing_calibration_returns_similarity_without_prediction(monkeypatch):
    import backend.api.routes as routes
    class Uncalibrated:
        calibration=None
        model_name="test model"
        def embed(self,image): return __import__("numpy").array([1.0,0.0],dtype="float32")
    monkeypatch.setattr(routes,"service",Uncalibrated())
    buffer=io.BytesIO();Image.new("RGB",(48,48),"black").save(buffer,format="JPEG"); payload=buffer.getvalue()
    with TestClient(app) as client:
        response=client.post("/api/compare",files={"image_a":("a.jpg",payload,"image/jpeg"),"image_b":("b.jpg",payload,"image/jpeg")})
    assert response.status_code==200
    assert response.json()["similarity"]==1.0
    assert response.json()["same_product"] is None
    assert response.json()["calibrated"] is False
