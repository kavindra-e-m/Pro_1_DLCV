import csv, hashlib
from pathlib import Path
import pytest
from PIL import Image
from backend.services.grouping import complete_link_groups
from backend.services.inference import matching_checkpoint_results, validate_index_metadata
from scripts.dataset_cleaning import clean

def test_complete_link_rejects_transitive_chain():
    # A-B and B-C pass, while A-C does not; all three must not be grouped.
    result=complete_link_groups(["A","B","C"],{(0,1):.91,(0,2):.54,(1,2):.89},.8)
    assert sorted(sorted(group["files"]) for group in result["groups"])==[["A","B"]]
    assert "C" in result["unmatched"]
    assert result["pairwise_scores"][1]["similarity"]==.54

def test_index_checkpoint_mismatch_is_rejected():
    with pytest.raises(ValueError,match="checkpoint_sha256"):
        validate_index_metadata({"checkpoint_sha256":"old"},{"checkpoint_sha256":"new"})

def test_stale_model_metrics_are_hidden(tmp_path):
    checkpoint=tmp_path/"model.pt";checkpoint.write_bytes(b"current checkpoint")
    report={"checkpoint":str(checkpoint),"checkpoint_sha256":"old"}
    assert matching_checkpoint_results([report])==[]

def test_cross_identity_duplicate_hashes_are_excluded(tmp_path):
    image=tmp_path/"same.png";Image.new("RGB",(48,48),"purple").save(image)
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    rows=[]
    for identity in ("a","b"):
        rows.append({"product_identity":identity,"image_path":str(image),"image_hash":digest,
            "image_width":"48","image_height":"48","source_dataset":"fixture"})
    manifest=tmp_path/"manifest.csv"
    with manifest.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    report=clean(manifest)
    with manifest.open(encoding="utf-8",newline="") as f: remaining=list(csv.DictReader(f))
    assert remaining==[]
    assert len(report["ambiguous_cross_identity_duplicate_groups_excluded"])==1
