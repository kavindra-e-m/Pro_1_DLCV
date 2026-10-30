import csv, json
from collections import Counter, defaultdict
from pathlib import Path
from ml.datasets.pairs import validate_image

ROOT = Path(__file__).resolve().parents[1]
def inspect(manifest=ROOT / "data/processed/metadata/catalog.csv"):
    rows = list(csv.DictReader(Path(manifest).open(encoding="utf-8", newline=""))) if Path(manifest).exists() else []
    identities, categories, sources = defaultdict(int), Counter(), Counter(); corrupt = 0; valid = 0
    for row in rows:
        ok, _, _ = validate_image(row["image_path"])
        if not ok: corrupt += 1; continue
        valid += 1; identities[row["product_identity"]] += 1
        categories[row.get("category") or "unspecified"] += 1
        sources[row.get("source_dataset") or "unspecified"] += 1
    positive_pairs = sum(n * (n - 1) // 2 for n in identities.values())
    identity_count = len(identities)
    quality_path=ROOT / "data/reports/data_quality_report.json"
    quality=json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    audit_path=ROOT / "data/reports/handbag_dataset_audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    report = {"manifest": str(manifest), "scope": audit.get("scope", "Bag-category ABO sample; refine before handbag-only claims"), "downloaded_images": len(rows), "valid_images": valid,
              "bag_product_identities": identity_count, "images_per_identity": dict(sorted(identities.items())),
              "identities_with_fewer_than_two_photographs": sorted(k for k,n in identities.items() if n < 2),
              "positive_pairs": positive_pairs, "negative_pairs_possible": sum(identities[a] * identities[b] for i, a in enumerate(identities) for b in list(identities)[i+1:]),
              "corrupted_images": corrupt, "duplicate_files_removed": len(quality.get("same_identity_duplicate_rows_removed",[])),
              "ambiguous_cross_identity_duplicate_groups_excluded": len(quality.get("ambiguous_cross_identity_duplicate_groups_excluded",[])),
              "category_distribution": dict(categories), "source_distribution": dict(sources),
              "handbag_audit": audit}
    out = ROOT / "data/reports/dataset_statistics.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report
if __name__ == "__main__": inspect()
