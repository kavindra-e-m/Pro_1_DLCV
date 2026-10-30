import csv, hashlib, json
from pathlib import Path
from ml.datasets.pairs import validate_image

ROOT = Path(__file__).resolve().parents[1]
def clean(manifest=ROOT / "data/processed/metadata/catalog.csv"):
    manifest = Path(manifest)
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f)); fields = list(rows[0]) if rows else []
    by_hash = {}; corrupted = []
    for row in rows:
        path = Path(row["image_path"]); ok, width, height = validate_image(path)
        if not ok: corrupted.append({"path": str(path), "reason": "decode_or_dimensions"}); continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row.update(image_hash=digest,image_width=width,image_height=height)
        by_hash.setdefault(digest, []).append(row)
    valid, within_identity, cross_identity = [], [], []
    for digest, group in by_hash.items():
        identities = {r["product_identity"] for r in group}
        if len(identities) > 1:
            cross_identity.append({"image_hash":digest,"product_identities":sorted(identities),"paths":[r["image_path"] for r in group]})
            continue
        valid.append(group[0])
        within_identity.extend(r["image_path"] for r in group[1:])
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(valid)
    report = {"input_images": len(rows), "valid_images": len(valid), "corrupted_images": corrupted,
              "same_identity_duplicate_rows_removed": within_identity,
              "ambiguous_cross_identity_duplicate_groups_excluded": cross_identity,
              "identities_present_before_cleaning": len({r["product_identity"] for r in rows}),
              "identities_present_after_cleaning": len({r["product_identity"] for r in valid})}
    out = ROOT / "data/reports/data_quality_report.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2)); return report
if __name__ == "__main__": clean()
