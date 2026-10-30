import csv, hashlib, json, logging, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

class ModelUnavailable(RuntimeError): pass

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def validate_index_metadata(metadata, expected):
    for key,value in expected.items():
        if metadata.get(key)!=value:
            raise ValueError(f"Stale or incompatible index: {key} does not match current artifacts.")

def matching_checkpoint_results(results):
    current=[]
    for result in results:
        path=Path(result.get("checkpoint",""))
        if path.is_file() and result.get("checkpoint_sha256")==sha256_file(path): current.append(result)
    return current

class InferenceService:
    def __init__(self):
        self.checkpoint_override=bool(os.getenv("BAGVISION_CHECKPOINT"))
        self.checkpoint=Path(os.getenv("BAGVISION_CHECKPOINT","artifacts/checkpoints/siamese_resnet50.pt"))
        self.manifest_path=Path(os.getenv("BAGVISION_CATALOG","data/processed/metadata/catalog.csv"))
        self.index_path=Path(os.getenv("BAGVISION_INDEX","artifacts/faiss_index/catalog.index"))
        self.model=None; self.transform=None; self.device="cpu"; self.catalog=[]; self.embeddings=None
        self.metrics={}; self.calibration=None; self.faiss_index=None; self.index_error="Index has not been loaded."
        self.selected_name="Siamese ResNet50"
        self.checkpoint_hash=None; self._load()

    def _load(self):
        for path,key in ((Path("artifacts/metrics/evaluation.json"),None),
                         (Path("artifacts/metrics/training.json"),"history"),
                         (Path("artifacts/metrics/triplet_training.json"),"triplet_history"),
                         (Path("artifacts/metrics/dinov2_retrieval.json"),"dinov2_retrieval")):
            if path.exists():
                try:
                    data=json.loads(path.read_text(encoding="utf-8"))
                    if key: self.metrics[key]=data
                    elif isinstance(data,dict): self.metrics.update(data)
                except (OSError,json.JSONDecodeError): logging.getLogger(__name__).exception("Invalid metrics file: %s",path)
        selected=self.metrics.get("selected_model") or {}
        if selected.get("model"): self.selected_name=selected["model"]
        if not self.checkpoint_override and selected.get("checkpoint"):
            candidate=Path(selected["checkpoint"])
            if candidate.is_file(): self.checkpoint=candidate
        if self.checkpoint.exists():
            try:
                import torch
                from torchvision import transforms
                from ml.models.siamese import SiameseNetwork
                self.device="cuda" if torch.cuda.is_available() else "cpu"
                state=torch.load(self.checkpoint,map_location=self.device,weights_only=False)
                self.checkpoint_hash=sha256_file(self.checkpoint)
                if selected.get("checkpoint_sha256") and selected.get("checkpoint_sha256")!=self.checkpoint_hash:
                    self.selected_name="Uncalibrated ResNet50 checkpoint"
                    self.metrics.pop("selected_model",None)
                self.model=SiameseNetwork(embedding_dim=int(state.get("embedding_dim",256)))
                self.model.load_state_dict(state["model_state_dict"]); self.model.to(self.device).eval()
                self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
                if self.metrics.get("models"):
                    current=matching_checkpoint_results(self.metrics["models"])
                    self.metrics["models"]=current; self.metrics["model_comparisons"]=current
                    siamese_result=next((r for r in current if Path(r["checkpoint"]).resolve()==self.checkpoint.resolve()),None)
                    if siamese_result: self.metrics["calibration"]=siamese_result.get("calibration")
                    else: self.metrics.pop("calibration",None)
                dino=self.metrics.get("dinov2_retrieval")
                if dino and dino.get("manifest_sha256")!=sha256_file(self.manifest_path):
                    self.metrics.pop("dinov2_retrieval",None)
                    self.metrics.pop("frozen_dinov2_retrieval",None)
                calibration=self.metrics.get("calibration")
                if calibration and calibration.get("checkpoint_sha256")==self.checkpoint_hash:
                    self.calibration=calibration
                else:
                    self.metrics.pop("calibration",None)
                self._load_index()
            except Exception as exc:
                self.model=None; logging.getLogger(__name__).exception("Could not load checkpoint: %s",exc)
        else:
            self.index_error="No trained checkpoint is available."

    def _load_index(self):
        try:
            import csv
            metadata_path=self.index_path.parent/"index_metadata.json"
            embeddings_path=Path("artifacts/embeddings/catalog.npy")
            catalog_path=Path("artifacts/embeddings/catalog.csv")
            if not metadata_path.is_file() or not embeddings_path.is_file() or not catalog_path.is_file():
                raise ValueError("Retrieval index files are missing; rebuild the index.")
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
            expected={"manifest_sha256":sha256_file(self.manifest_path),"checkpoint_sha256":self.checkpoint_hash,
                      "catalog_csv_sha256":sha256_file(catalog_path),"vectors_sha256":sha256_file(embeddings_path),
                      "index_file_sha256":sha256_file(self.index_path),"selected_model":self.selected_name}
            validate_index_metadata(metadata,expected)
            with catalog_path.open(encoding="utf-8",newline="") as f: catalog=list(csv.DictReader(f))
            with self.manifest_path.open(encoding="utf-8-sig",newline="") as f: manifest=list(csv.DictReader(f))
            matrix=np.load(embeddings_path,mmap_mode="r")
            if len(catalog)!=len(manifest) or len(catalog)!=int(metadata.get("count",-1)) or matrix.shape[0]!=len(catalog):
                raise ValueError("Manifest, retrieval catalog and vector row counts differ; rebuild the index.")
            if matrix.ndim!=2 or matrix.shape[1]!=int(metadata.get("dimension",-1)):
                raise ValueError("Embedding dimensions do not match index metadata.")
            if [(r.get("product_identity"),r.get("image_hash")) for r in catalog] != [(r.get("product_identity"),r.get("image_hash")) for r in manifest]:
                raise ValueError("Retrieval catalog row order/content differs from source manifest.")
            if metadata.get("ordered_product_identities")!=[r.get("product_identity") for r in catalog] or metadata.get("ordered_image_hashes")!=[r.get("image_hash") for r in catalog]:
                raise ValueError("Index row ordering metadata differs from the retrieval catalog.")
            if int(metadata["dimension"])!=int(state_dim:=self.model.embedding_dim):
                raise ValueError(f"Index dimension {metadata['dimension']} differs from checkpoint dimension {state_dim}.")
            self.catalog=catalog; self.embeddings=np.asarray(matrix)
            if metadata.get("backend")=="faiss.IndexFlatIP":
                try: import faiss
                except ImportError: self.faiss_index=None
                else:
                    index=faiss.read_index(str(self.index_path))
                    if int(index.ntotal)!=len(catalog) or int(index.d)!=matrix.shape[1]: raise ValueError("FAISS index dimensions or row count do not match its catalog.")
                    self.faiss_index=index
            elif metadata.get("backend")=="numpy_exact":
                self.faiss_index=None
            else:
                raise ValueError(f"Unsupported retrieval index backend: {metadata.get('backend')}.")
            self.index_error=None
        except Exception as exc:
            self.catalog=[]; self.embeddings=None; self.faiss_index=None; self.index_error=str(exc)
            logging.getLogger(__name__).error("Rejected retrieval index: %s",exc)

    @property
    def ready(self): return self.model is not None
    @property
    def model_name(self): return self.selected_name if self.ready else "Unavailable"
    @property
    def index_size(self): return len(self.catalog) if self.embeddings is not None and self.index_error is None else 0
    @property
    def indexed_at(self):
        try: return datetime.fromtimestamp(self.index_path.stat().st_mtime,timezone.utc).isoformat()
        except OSError: return None

    def embed(self,image:Image.Image):
        if not self.ready: raise ModelUnavailable("No trained checkpoint is available.")
        import torch
        tensor=self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode(): vector=self.model.embed(tensor)[0].cpu().numpy().astype("float32")
        return vector/(np.linalg.norm(vector)+1e-12)

    def search(self,vector,limit):
        if self.index_error: raise ModelUnavailable(self.index_error)
        if self.embeddings is None or not self.catalog: return []
        if self.faiss_index is not None:
            k=min(limit,int(self.faiss_index.ntotal)); scores,order=self.faiss_index.search(np.asarray(vector,dtype="float32").reshape(1,-1),k)
            ranked=((int(i),float(score)) for i,score in zip(order[0],scores[0]) if 0<=i<len(self.catalog))
        else:
            scores=self.embeddings@vector; order=np.argsort(-scores)[:limit]
            ranked=((int(i),float(scores[i])) for i in order)
        return [{**self.catalog[i],"similarity":score} for i,score in ranked]
