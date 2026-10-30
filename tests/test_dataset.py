"""Isolated unit fixtures. These generated PNGs are never included in project data or training."""
import csv
from pathlib import Path
from PIL import Image
from ml.datasets.pairs import PairDataset, validate_image
from scripts.dataset_split import split

def make_fixture(path: Path, color):
    Image.new("RGB", (48, 48), color).save(path, format="PNG")

def test_image_validation_and_identity_labeled_pairs(tmp_path):
    rows=[]
    for identity, color in [("unit-a", "red"), ("unit-b", "blue")]:
        for idx in range(2):
            path=tmp_path/f"{identity}-{idx}.png"; make_fixture(path,color)
            rows.append({"product_identity":identity,"image_path":str(path)})
    manifest=tmp_path/"manifest.csv"
    with manifest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["product_identity","image_path"]);writer.writeheader();writer.writerows(rows)
    dataset=PairDataset(manifest)
    assert len(dataset)>0
    assert {dataset.rows[i][3] for i in range(len(dataset))} == {0,1}
    assert all((Path(a).stem.rsplit("-",1)[0] == Path(b).stem.rsplit("-",1)[0]) == bool(same) for _,a,b,same in dataset.rows)
    assert validate_image(rows[0]["image_path"])[0]

def test_split_keeps_identity_groups_disjoint(tmp_path):
    rows=[]
    for i in range(10):
        for j in range(2): rows.append({"product_identity":f"id-{i}","image_path":f"{i}-{j}.jpg"})
    manifest=tmp_path/"all.csv"
    with manifest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    split(manifest,tmp_path/"splits",seed=23)
    groups=[]
    for name in ("train","validation","test"):
        with (tmp_path/"splits"/f"{name}.csv").open(encoding="utf-8",newline="") as f: groups.append({r["product_identity"] for r in csv.DictReader(f)})
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
