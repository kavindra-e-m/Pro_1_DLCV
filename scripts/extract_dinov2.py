"""Extract, but do not fine-tune, official pretrained DINOv2-base features."""
import argparse, csv, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from transformers import AutoImageProcessor, AutoModel

def retrieve_report(vectors, rows):
    identities=np.asarray([r["product_identity"] for r in rows]); scores=vectors@vectors.T
    recalls={1:[],5:[],10:[]}; aps=[]
    for i,identity in enumerate(identities):
        scores[i,i]=-np.inf; order=np.argsort(-scores[i]); order=order[order!=i]; labels=(identities[order]==identity).astype(int)
        if labels.sum()==0: continue
        for k in recalls: recalls[k].append(float(labels[:k].any()))
        aps.append(float(average_precision_score(labels,scores[i,order])))
    return {"model":"facebook/dinov2-base","fine_tuned":False,"queries_with_relevant_test_images":len(aps),
            "recall_at_1":float(np.mean(recalls[1])) if aps else None,"recall_at_5":float(np.mean(recalls[5])) if aps else None,
            "recall_at_10":float(np.mean(recalls[10])) if aps else None,"mean_average_precision":float(np.mean(aps)) if aps else None}

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",default="facebook/dinov2-base");p.add_argument("--batch-size",type=int,default=8);a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model_source=a.model
    if "/" in a.model:
        owner,repo=a.model.split("/",1);snapshots=Path.home()/".cache/huggingface/hub"/f"models--{owner}--{repo}"/"snapshots"
        cached=next((p for p in sorted(snapshots.glob("*")) if (p/"config.json").exists() and (p/"model.safetensors").exists()),None)
        if cached is not None: model_source=str(cached)
    processor=AutoImageProcessor.from_pretrained(model_source);model=AutoModel.from_pretrained(model_source).to(device).eval()
    with open("data/splits/test.csv",encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    vectors=[];kept=[]; elapsed=0.
    for start in range(0,len(rows),a.batch_size):
        batch_rows=rows[start:start+a.batch_size];images=[];batch_kept=[]
        for row in batch_rows:
            try:
                with Image.open(row["image_path"]) as im: images.append(im.convert("RGB"))
                batch_kept.append(row)
            except OSError as exc: print(f"skip {row['image_path']}: {exc}")
        if not images: continue
        started=time.perf_counter();inputs=processor(images=images,return_tensors="pt").to(device)
        with torch.inference_mode(): features=model(**inputs).last_hidden_state[:,0]
        if device.type=="cuda":torch.cuda.synchronize()
        elapsed+=time.perf_counter()-started
        vectors.extend(torch.nn.functional.normalize(features,dim=1).cpu().numpy());kept.extend(batch_kept)
    if not vectors: raise RuntimeError("The test split contains no readable images.")
    matrix=np.asarray(vectors,dtype="float32");report=retrieve_report(matrix.copy(),kept)
    model_dir=Path(model_source); model_files=list(model_dir.glob("*.safetensors"))+list(model_dir.glob("pytorch_model*.bin")) if model_dir.is_dir() else []
    report.update({"device":str(device),"test_images":len(kept),"pretrained_model_id":a.model,
        "manifest_sha256":sha256_file("data/processed/metadata/catalog.csv"),
        "inference_latency_ms_per_image":elapsed*1000/max(len(kept),1),"model_size_bytes":sum(x.stat().st_size for x in model_files),
        "test_identities":len({r["product_identity"] for r in kept}),"timed_wall_seconds":elapsed})
    out=Path("artifacts/metrics");out.mkdir(parents=True,exist_ok=True);(out/"dinov2_retrieval.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    emb=Path("artifacts/embeddings");emb.mkdir(parents=True,exist_ok=True);np.save(emb/"dinov2_test.npy",matrix)
    with (emb/"dinov2_test.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=kept[0].keys());writer.writeheader();writer.writerows(kept)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
