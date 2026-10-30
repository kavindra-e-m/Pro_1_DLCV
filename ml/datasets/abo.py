"""Focused acquisition and validation helpers for the official Amazon Berkeley Objects release."""
from __future__ import annotations
import csv, gzip, hashlib, json, re, shutil, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/abo"
BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com"
TERMS = re.compile(r"\b(handbag|shoulder bag|tote bag|purse|backpack|satchel|clutch|crossbody|messenger bag|duffel bag|luggage|suitcase|briefcase|diaper bag|wallet|camera bag)\b", re.I)
CATEGORY_TERMS = re.compile(r"\b(bags?|luggage|suitcases?)\b", re.I)
BAG_TYPES = {"HANDBAG", "BACKPACK", "BAG", "LUGGAGE", "SUITCASE", "BRIEFCASE", "WALLET", "CAMERA_BAG", "CAMERA_BAGS_AND_CASES", "TRAVEL_BAG", "DUFFEL_BAG", "TOTE_BAG", "CROSSBODY_BAG"}

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size: return
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "BagVision-AI/1.0 research dataset acquisition"})
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(dest)

def fields(record: dict, key: str) -> str:
    value = record.get(key, [])
    if isinstance(value, str): return value
    if isinstance(value, list): return " ".join(str(v.get("value", "")) if isinstance(v, dict) else str(v) for v in value)
    return ""

def is_bag(record: dict) -> bool:
    product_type = fields(record, "product_type").upper().replace(" ", "_")
    if product_type: return product_type in BAG_TYPES
    # Use category and title text only; long descriptions/keywords contain unrelated incidental mentions.
    category = " ".join(str(x.get("path", "")) for x in record.get("node", []) if isinstance(x, dict))
    title = fields(record, "item_name")
    return bool(CATEGORY_TERMS.search(category) or TERMS.search(title))

def acquire(max_products: int | None = None, max_images_per_product: int | None = None) -> dict:
    """Download and inspect only matching bag listings and their actual catalog images."""
    listings_archive = RAW / "abo-listings.tar"
    download(f"{BASE}/archives/abo-listings.tar", listings_archive)
    listing_root = RAW / "listings"
    if not list((listing_root / "metadata").glob("listings_*.json.gz")):
        with tarfile.open(listings_archive) as tf:
            for member in tf.getmembers():
                if member.name.startswith("listings/") or member.name == "LICENSE-CC-BY-4.0.txt": tf.extract(member, RAW, filter="data")
    products = []
    for shard in sorted((listing_root / "metadata").glob("listings_*.json.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as src:
            for line in src:
                try: rec = json.loads(line)
                except json.JSONDecodeError: continue
                if rec.get("item_id") and is_bag(rec): products.append(rec)
    products.sort(key=lambda x: (str(x.get("domain_name", "")), str(x.get("item_id", ""))))
    if max_products: products = products[:max_products]
    meta = RAW / "images-metadata.csv.gz"
    download(f"{BASE}/images/metadata/images.csv.gz", meta)
    wanted = {}
    for product in products:
        ids = [product.get("main_image_id"), *(product.get("other_image_id") or [])]
        if max_images_per_product: ids = ids[:max_images_per_product]
        for image_id in ids:
            if image_id: wanted.setdefault(str(image_id), []).append(product)
    image_meta = {}
    with gzip.open(meta, "rt", encoding="utf-8", newline="") as src:
        for row in csv.DictReader(src):
            if row["image_id"] in wanted: image_meta[row["image_id"]] = row
    out_dir = ROOT / "data/processed/images/abo"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/processed/metadata/catalog.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source_dataset", "source_product_id", "product_identity", "category", "brand", "image_path", "source_url", "original_filename", "image_hash", "image_width", "image_height", "license"]
    errors = []; rows = []
    def fetch_one(image_id, products_for_image):
        info = image_meta.get(image_id)
        if not info: return None, {"image_id": image_id, "reason": "missing_image_metadata"}
        path = info["path"]; ext = Path(path).suffix
        url = f"{BASE}/images/small/{path}"
        source_copy = RAW / "image-cache" / f"{image_id}{ext}"
        target = out_dir / f"{image_id}.jpg"
        try:
            if target.exists():
                with Image.open(target) as im: im.verify()
                with Image.open(target) as im: width, height = im.size
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
            else:
                download(url, source_copy)
                with Image.open(source_copy) as im: im.verify()
                with Image.open(source_copy) as im:
                    width, height = im.size; im.convert("RGB").save(target, format="JPEG", quality=95)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
            image_rows = []
            for product in products_for_image:
                domain = str(product.get("domain_name", "unknown")); item_id = str(product["item_id"])
                image_rows.append({"source_dataset": "ABO", "source_product_id": f"{domain}:{item_id}",
                     "product_identity": f"ABO:{domain}:{item_id}", "category": fields(product, "product_type"),
                     "brand": fields(product, "brand"), "image_path": str(target.resolve()), "source_url": url,
                     "original_filename": f"{image_id}{ext}", "image_hash": digest, "image_width": width,
                     "image_height": height, "license": "CC BY 4.0"})
            return image_rows, None
        except Exception as exc:
            return None, {"image_id": image_id, "reason": str(exc)}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch_one, image_id, products_for_image) for image_id, products_for_image in wanted.items()]
        for future in as_completed(futures):
            row, error = future.result()
            if error: errors.append(error)
            elif row: rows.extend(row)
    with manifest.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=cols); writer.writeheader(); writer.writerows(rows)
    report = {"source": "Amazon Berkeley Objects", "source_url": "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html",
              "license": "CC BY 4.0", "attribution": "Amazon.com; ABO dataset authors listed in docs/dataset_documentation.md",
              "category_filter": {"product_types": sorted(BAG_TYPES), "title_or_category_pattern": TERMS.pattern}, "matching_listing_records": len(products), "candidate_images": sum(map(len,wanted.values())),
              "image_metadata_matches": len(image_meta), "downloaded_valid_images": len(rows),
              "identities_with_images": len({r['product_identity'] for r in rows}), "failed_images": len(errors),
              "errors": errors, "manifest": str(manifest)}
    report_path = ROOT / "data/reports/dataset_acquisition.json"; report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
