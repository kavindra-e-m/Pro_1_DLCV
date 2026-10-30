import csv, json
from pathlib import Path
from backend.database.db import SessionLocal, init_db
from backend.database.models import Image, Product
from backend.services.inference import sha256_file

def import_catalog(manifest="data/processed/metadata/catalog.csv"):
    init_db()
    with Path(manifest).open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    db=SessionLocal(); added_images=0
    try:
        # This generated local catalog is a rebuildable cache; clear only ABO rows before resync.
        old_ids=[p.id for p in db.query(Product.id).filter_by(source_dataset="ABO").all()]
        if old_ids:
            db.query(Image).filter(Image.product_id.in_(old_ids)).delete(synchronize_session=False)
            db.query(Product).filter(Product.id.in_(old_ids)).delete(synchronize_session=False)
        products={}
        for row in rows:
            key=(row["source_dataset"],row["source_product_id"])
            product=products.get(key) or db.query(Product).filter_by(source_dataset=key[0],source_product_id=key[1]).first()
            if product is None:
                product=Product(source_dataset=key[0],source_product_id=key[1],product_identity=row["product_identity"],category=row.get("category") or None,brand=row.get("brand") or None,metadata_json=json.dumps({"license":row.get("license")}));db.add(product);db.flush()
            products[key]=product
            exists=db.query(Image).filter_by(image_hash=row["image_hash"]).first()
            if exists is None:
                db.add(Image(product_id=product.id,image_path=row["image_path"],original_filename=row["original_filename"],image_hash=row["image_hash"],source_url=row["source_url"],image_width=int(row["image_width"]),image_height=int(row["image_height"])));added_images+=1
        db.commit()
    except Exception:
        db.rollback();raise
    finally: db.close()
    marker=Path("backend/data/catalog_import_metadata.json"); marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({"manifest":str(Path(manifest).resolve()),"manifest_sha256":sha256_file(manifest),"images":len(rows)},indent=2),encoding="utf-8")
    print(f"Imported {len(rows)-added_images} existing and {added_images} new image records.")
    return {"images":len(rows),"new_images":added_images}
if __name__=="__main__":import_catalog()
