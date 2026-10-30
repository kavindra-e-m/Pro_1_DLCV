"""Enrich the acquired ABO manifest from source listings and audit a narrow handbag set."""
import csv, gzip, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

from ml.datasets.handbag_filter import is_target_handbag, normalize
from ml.datasets.pairs import validate_image
from scripts.dataset_split import split

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/metadata/catalog.csv"
LISTINGS = ROOT / "data/raw/abo/listings/metadata"

def load_listing_records():
    records = {}
    for shard in sorted(LISTINGS.glob("listings_*.json.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as src:
            for line in src:
                try: record = json.loads(line)
                except json.JSONDecodeError: continue
                if record.get("item_id"):
                    records[f"{record.get('domain_name','unknown')}:{record['item_id']}"] = record
    return records

def refine(manifest=MANIFEST, seed=42, output_manifest=None, snapshot=False):
    source_manifest=Path(manifest); manifest=Path(output_manifest) if output_manifest else source_manifest
    with source_manifest.open(encoding="utf-8-sig", newline="") as src:
        input_rows = list(csv.DictReader(src))
    original = manifest.with_name("catalog_unfiltered.csv")
    if source_manifest.resolve()!=original.resolve() and (snapshot or not original.exists()):
        with original.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(input_rows[0]) if input_rows else ["product_identity", "image_path"])
            writer.writeheader(); writer.writerows(input_rows)
    source_records = load_listing_records()
    retained, excluded, audit_rows = [], Counter(), []
    for row in input_rows:
        record = source_records.get(row.get("source_product_id", ""), {})
        source_type = normalize(record.get("product_type")) or row.get("category", "")
        title = normalize(record.get("item_name"))
        category_paths = " | ".join(str(n.get("path", "")) for n in record.get("node", []) if isinstance(n, dict))
        if not record:
            # Existing manifest category is source metadata; without the matching source
            # listing we retain only the unambiguous HANDBAG type.
            source_type = row.get("category", "")
        row["category"] = source_type
        row["source_item_name"] = title
        row["source_category_paths"] = category_paths
        row["source_product_type"] = source_type
        if is_target_handbag(source_type, title, category_paths):
            retained.append(row)
        else:
            excluded[source_type or "UNKNOWN"] += 1
    valid = []; invalid_image_count=0
    for row in retained:
        ok, width, height = validate_image(row["image_path"])
        if not ok:
            excluded[f"INVALID_IMAGE:{row.get('category') or 'UNKNOWN'}"] += 1
            invalid_image_count+=1
            continue
        digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        row.update(image_hash=digest, image_width=str(width), image_height=str(height))
        valid.append(row)
    hashes = defaultdict(list)
    for row in valid: hashes[row["image_hash"]].append(row)
    duplicate_hashes = []
    kept = []
    for digest, group in hashes.items():
        ids = sorted({r["product_identity"] for r in group})
        if len(ids) > 1:
            duplicate_hashes.append({"sha256": digest, "product_identities": ids, "row_count": len(group), "action": "excluded_all_ambiguous_rows"})
        else:
            kept.append(group[0])
            if len(group) > 1:
                duplicate_hashes.append({"sha256": digest, "product_identities": ids, "row_count": len(group), "action": "retained_one_same_identity_image"})
    identity_counts = Counter(r["product_identity"] for r in kept)
    categories = Counter(r.get("category") or "UNSPECIFIED" for r in kept)
    category_ids = defaultdict(set)
    for r in kept: category_ids[r.get("category") or "UNSPECIFIED"].add(r["product_identity"])
    fields = list(kept[0]) if kept else list(input_rows[0])
    with manifest.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader(); writer.writerows(kept)
    split(manifest, ROOT / "data/splits", seed)
    split_ids = {}
    for name in ("train", "validation", "test"):
        with (ROOT / f"data/splits/{name}.csv").open(encoding="utf-8", newline="") as src:
            split_ids[name] = {r["product_identity"] for r in csv.DictReader(src)}
    overlaps = {"train_validation": sorted(split_ids["train"] & split_ids["validation"]),
                "train_test": sorted(split_ids["train"] & split_ids["test"]),
                "validation_test": sorted(split_ids["validation"] & split_ids["test"])}
    report = {
        "scope": "ABO handbags, purses, fashion totes, shoulder bags, clutches and crossbody bags; source product type/title evidence required",
        "source_manifest": str(source_manifest), "refined_manifest": str(manifest),
        "input_rows": len(input_rows), "retained_product_identities": len(identity_counts),
        "retained_images": len(kept), "valid_images": len(kept),"invalid_image_count":invalid_image_count,
        "categories_retained_image_counts": dict(categories),
        "categories_retained_identity_counts": {k: len(v) for k, v in category_ids.items()},
        "categories_excluded_image_counts": dict(excluded),
        "images_per_identity": dict(sorted(identity_counts.items())),
        "identities_with_fewer_than_two_photographs": sorted(k for k, n in identity_counts.items() if n < 2),
        "duplicate_image_hashes": duplicate_hashes,
        "identity_split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "identity_split_overlaps": overlaps,
        "identity_splits_disjoint": not any(overlaps.values()),
        "identity_source_note": "Identities are ABO domain_name:item_id product records; images are catalog images and are not asserted to be independent real-world photographs.",
        "seed": seed,
    }
    out = ROOT / "data/reports/handbag_dataset_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    from scripts.write_handbag_audit import write_report
    write_report(out)
    print(json.dumps(report, indent=2)); return report

if __name__ == "__main__": refine()
