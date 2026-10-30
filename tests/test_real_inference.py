"""Optional integration test over the locally acquired ABO images and trained checkpoint."""
import csv
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.main import app

MANIFEST=Path("data/processed/metadata/catalog.csv")
CHECKPOINT=Path("artifacts/checkpoints/siamese_resnet50.pt")
INDEX=Path("artifacts/embeddings/catalog.npy")
pytestmark=pytest.mark.skipif(not (MANIFEST.exists() and CHECKPOINT.exists() and INDEX.exists()),reason="Real ABO data, checkpoint and index are required for this integration test.")

def test_real_image_search_compare_and_duplicate_detection():
    with MANIFEST.open(encoding="utf-8",newline="") as f: image_path=Path(next(csv.DictReader(f))["image_path"])
    image_bytes=image_path.read_bytes()
    with TestClient(app) as client:
        dashboard=client.get("/api/dashboard").json()
        assert dashboard["total_images"] >= 1
        assert dashboard["product_identities"] >= 1
        search=client.post("/api/search",files={"file":(image_path.name,image_bytes,"image/jpeg")})
        assert search.status_code==200
        assert search.json()["results"][0]["similarity"]>.99
        compare=client.post("/api/compare",files={"image_a":(image_path.name,image_bytes,"image/jpeg"),"image_b":(image_path.name,image_bytes,"image/jpeg")})
        assert compare.status_code==200 and compare.json()["similarity"]>.99
        duplicates=client.post("/api/duplicate-detection",files=[("files",(image_path.name,image_bytes,"image/jpeg")),("files",(image_path.name,image_bytes,"image/jpeg"))])
        assert duplicates.status_code==200 and len(duplicates.json()["groups"])==1
