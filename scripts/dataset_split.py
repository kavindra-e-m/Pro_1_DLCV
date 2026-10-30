import argparse, csv, random
from collections import defaultdict
from pathlib import Path

def split(manifest, output, seed=42, ratios=(.8,.1,.1)):
    with Path(manifest).open(encoding="utf-8", newline="") as f: rows=list(csv.DictReader(f))
    by_identity=defaultdict(list)
    for r in rows: by_identity[r["product_identity"]].append(r)
    keys=sorted(by_identity); random.Random(seed).shuffle(keys)
    n=len(keys); a=int(n*ratios[0]); b=int(n*(ratios[0]+ratios[1]))
    if n and a == 0: a = 1
    if n > 1 and b <= a: b = a + 1
    partitions={"train":keys[:a],"validation":keys[a:b],"test":keys[b:]}
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    for name, ids in partitions.items():
        subset=[r for key in ids for r in by_identity[key]]
        with (out/f"{name}.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["product_identity","image_path"]); w.writeheader(); w.writerows(subset)
        print(f"{name}: {len(subset)} images, {len(ids)} identities")
    assert not (set(partitions["train"]) & set(partitions["validation"]))
    assert not (set(partitions["train"]) & set(partitions["test"]))
    assert not (set(partitions["validation"]) & set(partitions["test"]))
    return partitions
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",default="data/processed/metadata/catalog.csv");p.add_argument("--output",default="data/splits");p.add_argument("--seed",type=int,default=42);a=p.parse_args();split(a.manifest,a.output,a.seed)
