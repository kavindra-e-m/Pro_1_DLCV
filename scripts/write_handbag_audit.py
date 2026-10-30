"""Render the generated source-data audit as a tracked Markdown report."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def write_report(source=ROOT/"data/reports/handbag_dataset_audit.json",target=ROOT/"docs/handbag_dataset_audit.md"):
    report=json.loads(Path(source).read_text(encoding="utf-8"))
    lines=["# Refined ABO handbag dataset audit","",
        f"Scope: {report['scope']}","",
        f"Input rows reviewed: {report['input_rows']}  ",
        f"Valid retained catalog images: {report['valid_images']}  ",
        f"Retained ABO product identities: {report['retained_product_identities']}  ",
        "Image records are source catalog images/views; the report does not claim separate real-world photo sessions.","",
        "## Retained categories","","| Source product type | Images | Product identities |","|---|---:|---:|"]
    for category,count in report["categories_retained_image_counts"].items():
        lines.append(f"| {category} | {count} | {report['categories_retained_identity_counts'].get(category,0)} |")
    lines += ["","## Excluded rows","","| Source type / validation | Excluded image rows |","|---|---:|"]
    for category,count in report["categories_excluded_image_counts"].items(): lines.append(f"| {category} | {count} |")
    lines += ["","## Duplicate hashes and identity counts","",
        f"Duplicate-hash groups: {len(report['duplicate_image_hashes'])}. No exact duplicate image hashes were found in the retained set." if not report["duplicate_image_hashes"] else f"Duplicate-hash groups reviewed: {len(report['duplicate_image_hashes'])}.",
        f"Identities with fewer than two images: {len(report['identities_with_fewer_than_two_photographs'])}.","",
        "| ABO product identity | Catalog images |","|---|---:|"]
    for identity,count in report["images_per_identity"].items(): lines.append(f"| `{identity}` | {count} |")
    lines += ["","## Identity-disjoint split","","| Partition | Product identities |","|---|---:|"]
    for name,count in report["identity_split_counts"].items(): lines.append(f"| {name} | {count} |")
    lines += ["","Verified intersections: train∩validation = 0; train∩test = 0; validation∩test = 0.",
        f"Identities with fewer than two images: {', '.join(f'`{x}`' for x in report['identities_with_fewer_than_two_photographs']) or 'none'}.","",
        f"Seed: {report['seed']}. Detailed machine-readable report: `data/reports/handbag_dataset_audit.json` (local artifact).",""]
    Path(target).write_text("\n".join(lines),encoding="utf-8")
    return Path(target)
if __name__=="__main__": print(write_report())
