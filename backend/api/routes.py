import csv, hashlib, io, json, logging, os
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image as PILImage, UnidentifiedImageError
from sqlalchemy import func
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database.models import Image, Product
from backend.services.inference import InferenceService, ModelUnavailable, sha256_file
from backend.services.grouping import complete_link_groups

router = APIRouter()
log = logging.getLogger("bagvision.api")
MAX_BYTES = int(os.getenv("BAGVISION_MAX_UPLOAD_MB", "15")) * 1024 * 1024
UPLOADS = Path("backend/uploads"); UPLOADS.mkdir(parents=True, exist_ok=True)
service = InferenceService()

async def read_image(file: UploadFile) -> tuple[bytes, PILImage.Image]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, WebP, or BMP image.")
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES: raise HTTPException(413, f"Image exceeds {MAX_BYTES // (1024*1024)} MB.")
    try:
        image = PILImage.open(io.BytesIO(content)).convert("RGB")
        if image.width < 32 or image.height < 32: raise ValueError("Image must be at least 32 by 32 pixels.")
        return content, image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

def embed(image: PILImage.Image):
    try: return service.embed(image)
    except ModelUnavailable as exc: raise HTTPException(503, str(exc)) from exc

@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.ready, "device": service.device,
            "dataset_indexed": service.index_size,"index_valid":service.index_error is None,
            "index_error":service.index_error,"calibrated":service.calibration is not None}

def catalog_consistency(db: Session):
    manifest=service.manifest_path
    if not manifest.is_file(): return {"consistent":False,"reason":"Source manifest is missing."}
    actual=sha256_file(manifest)
    marker=Path("backend/data/catalog_import_metadata.json")
    if not marker.is_file(): return {"consistent":False,"reason":"Database catalog has not been synchronized from the current manifest."}
    try: imported=json.loads(marker.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return {"consistent":False,"reason":"Database import metadata is invalid."}
    db_rows=db.query(Product.product_identity,Image.image_hash).join(Image,Image.product_id==Product.id).filter(Product.source_dataset=="ABO").all()
    index_rows={(r.get("product_identity"),r.get("image_hash")) for r in service.catalog}
    db_pairs={(identity,digest) for identity,digest in db_rows}
    consistent=(imported.get("manifest_sha256")==actual and db_pairs==index_rows and service.index_error is None)
    return {"consistent":consistent,"database_manifest_matches":imported.get("manifest_sha256")==actual,
            "database_image_rows":len(db_pairs),"retrieval_rows":len(index_rows),
            "reason":None if consistent else "Database, manifest and retrieval index are out of sync; rerun preparation/import and rebuild the index."}

@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)):
    return {"total_images": db.query(func.count(Image.id)).scalar() or 0,
            "product_identities": db.query(func.count(Product.id)).scalar() or 0,
            "model": service.model_name if service.ready else "No checkpoint configured",
            "indexed_images": service.index_size, "last_indexed": service.indexed_at,
            "metrics": service.metrics,"catalog_consistency":catalog_consistency(db),
            "index_error":service.index_error,"calibration":service.calibration}

@router.get("/model/metrics")
def model_metrics(): return service.metrics

@router.get("/dataset/statistics")
def dataset_statistics():
    path = Path("data/reports/dataset_statistics.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_prepared", "message": "Run scripts/prepare_data.py after acquiring a supported dataset."}

@router.get("/products")
def products(q: str = "", limit: int = 60, offset: int = 0, db: Session = Depends(get_db)):
    query = db.query(Product)
    if q: query = query.filter((Product.product_identity.ilike(f"%{q}%")) | (Product.category.ilike(f"%{q}%")) | (Product.brand.ilike(f"%{q}%")))
    rows = query.order_by(Product.id).offset(max(offset,0)).limit(max(1,min(limit,200))).all()
    return [{"id": p.id, "source_dataset": p.source_dataset, "source_product_id": p.source_product_id,
             "product_identity": p.product_identity, "category": p.category, "brand": p.brand,
             "images": [{"id": i.id, "url": "/media/abo/" + Path(i.image_path).name, "width": i.image_width, "height": i.image_height} for i in p.images]} for p in rows]

@router.get("/products/{product_id}")
def product(product_id: int, db: Session = Depends(get_db)):
    row = db.query(Product).filter(Product.id == product_id).first()
    if not row: raise HTTPException(404, "Product not found.")
    return {"id": row.id, "source_dataset": row.source_dataset, "source_product_id": row.source_product_id,
            "product_identity": row.product_identity, "category": row.category, "brand": row.brand,
            "metadata": json.loads(row.metadata_json), "images": [{"id": i.id, "url": "/media/abo/" + Path(i.image_path).name} for i in row.images]}

@router.post("/search")
async def search(file: Annotated[UploadFile, File()], limit: int = 10, db: Session = Depends(get_db)):
    consistency=catalog_consistency(db)
    if not consistency["consistent"]:
        raise HTTPException(503,consistency.get("reason") or "Database and retrieval catalog are not synchronized.")
    _, image = await read_image(file); vector = embed(image)
    try: results = service.search(vector, max(1, min(limit, 100)))
    except ModelUnavailable as exc: raise HTTPException(503,str(exc)) from exc
    for row in results:
        if row.get("image_path"): row["image_path"] = "/media/abo/" + Path(row["image_path"]).name
    return {"query_filename": file.filename, "results": results, "model": service.model_name,
            "catalog_consistency":consistency}

@router.post("/compare")
async def compare(image_a: Annotated[UploadFile, File()], image_b: Annotated[UploadFile, File()]):
    _, a = await read_image(image_a); _, b = await read_image(image_b)
    va, vb = embed(a), embed(b); score = float((va * vb).sum())
    calibration=service.calibration
    threshold=float(calibration["threshold"]) if calibration else None
    return {"similarity":score,"same_product":(score>=threshold) if threshold is not None else None,
            "calibrated":threshold is not None,"decision_status":"calibrated" if threshold is not None else "uncalibrated_similarity_only",
            "threshold":threshold,"calibration":calibration,"model":service.model_name,
            "explanation":"Cosine similarity of L2-normalized embeddings. A same/different decision is shown only when a checkpoint-matched validation calibration is available."}

@router.post("/duplicate-detection")
async def duplicate_detection(files: list[UploadFile] = File(...)):
    if len(files) < 2 or len(files) > 100: raise HTTPException(400, "Upload between 2 and 100 images.")
    images = [await read_image(f) for f in files]; vectors = [embed(image) for _, image in images]
    names=[f.filename or f"upload_{i}" for i,f in enumerate(files)]
    scores={}
    for i in range(len(vectors)):
        for j in range(i+1,len(vectors)): scores[(i,j)]=float((vectors[i]*vectors[j]).sum())
    if service.calibration:
        result=complete_link_groups(names,scores,float(service.calibration["threshold"]))
        return {**result,"calibrated":True,"threshold":service.calibration["threshold"],"calibration":service.calibration}
    pairs=[{"a":names[i],"b":names[j],"similarity":score,"above_threshold":None} for (i,j),score in scores.items()]
    return {"groups":[],"unmatched":names,"pairwise_scores":pairs,"manual_review_pairs":pairs,
            "grouping_policy":"uncalibrated_pairwise_review_only","calibrated":False,"threshold":None,"calibration":None}
