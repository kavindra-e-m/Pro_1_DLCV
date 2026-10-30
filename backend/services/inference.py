import json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

class ModelUnavailable(RuntimeError): pass

class InferenceService:
    def __init__(self):
        self.checkpoint = Path(os.getenv("BAGVISION_CHECKPOINT", "artifacts/checkpoints/siamese_resnet50.pt"))
        self.catalog_path = Path(os.getenv("BAGVISION_CATALOG", "data/processed/metadata/catalog.csv"))
        self.index_path = Path(os.getenv("BAGVISION_INDEX", "artifacts/faiss_index/catalog.index"))
        self.model = None; self.transform = None; self.device = "cpu"; self.catalog = []; self.embeddings = None; self.metrics = {}
        self.faiss_index = None
        self._load()

    def _load(self):
        metrics = Path("artifacts/metrics/evaluation.json")
        if metrics.exists(): self.metrics = json.loads(metrics.read_text(encoding="utf-8"))
        training = Path("artifacts/metrics/training.json")
        if training.exists(): self.metrics["history"] = json.loads(training.read_text(encoding="utf-8"))
        dino = Path("artifacts/metrics/dinov2_retrieval.json")
        if dino.exists(): self.metrics["dinov2_retrieval"] = json.loads(dino.read_text(encoding="utf-8"))
        if not self.checkpoint.exists(): return
        try:
            import torch
            from torchvision import transforms
            from ml.models.siamese import SiameseNetwork
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            self.model = SiameseNetwork(embedding_dim=int(state.get("embedding_dim", 256)))
            self.model.load_state_dict(state["model_state_dict"]); self.model.to(self.device).eval()
            self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            catalog_path = Path("artifacts/embeddings/catalog.csv")
            if catalog_path.exists():
                import csv
                with catalog_path.open(encoding="utf-8", newline="") as f: self.catalog = list(csv.DictReader(f))
                vectors_path = Path("artifacts/embeddings/catalog.npy")
                if vectors_path.exists(): self.embeddings = np.load(vectors_path)
            if self.index_path.exists():
                try:
                    import faiss
                    self.faiss_index = faiss.read_index(str(self.index_path))
                except (ImportError, RuntimeError):
                    self.faiss_index = None
        except Exception as exc:
            self.model = None
            import logging; logging.getLogger(__name__).exception("Could not load checkpoint: %s", exc)

    @property
    def ready(self): return self.model is not None
    @property
    def model_name(self): return "Siamese ResNet50" if self.ready else "Unavailable"
    @property
    def index_size(self): return len(self.catalog) if self.embeddings is not None else 0
    @property
    def indexed_at(self):
        try: return datetime.fromtimestamp(self.index_path.stat().st_mtime, timezone.utc).isoformat()
        except OSError: return None

    def embed(self, image: Image.Image):
        if not self.ready: raise ModelUnavailable("No trained checkpoint found. Train a model with `python scripts/train.py` and rebuild the index.")
        import torch
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode(): vector = self.model.embed(tensor)[0].cpu().numpy().astype("float32")
        return vector / (np.linalg.norm(vector) + 1e-12)

    def search(self, vector, limit):
        if self.embeddings is None or not self.catalog: return []
        if self.faiss_index is not None:
            k=min(limit,int(self.faiss_index.ntotal))
            scores, order = self.faiss_index.search(np.asarray(vector,dtype="float32").reshape(1,-1),k)
            scores, order = scores[0], order[0]
            ranked = ((int(i), float(score)) for i, score in zip(order, scores) if 0 <= i < len(self.catalog))
        else:
            scores = self.embeddings @ vector
            order = np.argsort(-scores)[:limit]
            ranked = ((int(i), float(scores[i])) for i in order)
        return [{**self.catalog[i], "similarity": score} for i, score in ranked]
