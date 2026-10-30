import csv
from collections import Counter
from pathlib import Path

def visualize(manifest="data/processed/metadata/catalog.csv", output="data/reports/category_distribution.png"):
    import matplotlib.pyplot as plt
    with Path(manifest).open(encoding="utf-8", newline="") as f: rows=list(csv.DictReader(f))
    counts=Counter(r.get("category") or "unspecified" for r in rows)
    labels, values=zip(*counts.most_common(20)) if counts else ([], [])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6)); ax.barh(labels[::-1],values[::-1],color="#cf6954"); ax.set_xlabel("Valid images"); ax.set_title("ABO bag category distribution (observed subset)"); fig.tight_layout(); fig.savefig(output,dpi=160); plt.close(fig)
    print(output)
if __name__ == "__main__": visualize()
