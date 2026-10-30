"""Validation-calibrated pair classification and identity-held-out retrieval evaluation."""
import csv, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from ml.datasets.pairs import PairDataset
from ml.models.siamese import SiameseNetwork, TripletNetwork

EVAL_TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def pair_scores(model,manifest,device):
    dataset=PairDataset(manifest,EVAL_TF,seed=42,negative_mode="random")
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    labels=[]; scores=[]; model.eval()
    with torch.inference_mode():
        for a,b,y in loader:
            ea,eb=model.embed(a.to(device)),model.embed(b.to(device))
            scores.extend((ea*eb).sum(1).cpu().numpy().tolist()); labels.extend(y.numpy().tolist())
    return np.asarray(labels),np.asarray(scores)

def select_threshold(labels,scores):
    candidates=np.unique(scores)
    choices=[]
    for threshold in candidates:
        pred=scores>=threshold
        choices.append((f1_score(labels,pred,zero_division=0),precision_score(labels,pred,zero_division=0),float(threshold)))
    return max(choices,key=lambda row:(row[0],row[1],row[2]))[2]

def retrieval_metrics(model,manifest,device):
    with Path(manifest).open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    vectors=[]; kept=[]; model.eval()
    with torch.inference_mode():
        for row in rows:
            try:
                with Image.open(row["image_path"]) as im: tensor=EVAL_TF(im.convert("RGB")).unsqueeze(0).to(device)
            except (OSError,ValueError): continue
            vectors.append(model.embed(tensor)[0].cpu().numpy()); kept.append(row)
    if not vectors: return {"test_images":0,"queries_with_relevant_test_images":0,"recall_at_1":None,"recall_at_5":None,"recall_at_10":None,"mean_average_precision":None}
    matrix=np.asarray(vectors); ids=np.asarray([r["product_identity"] for r in kept]); recalls={1:[],5:[],10:[]}; aps=[]
    for i,identity in enumerate(ids):
        scores=matrix@matrix[i]; scores[i]=-np.inf; order=np.argsort(-scores); relevant=(ids[order]==identity)
        if not relevant.any(): continue
        for k in recalls: recalls[k].append(float(relevant[:k].any()))
        ranks=np.flatnonzero(relevant)+1; aps.append(float(np.mean(np.arange(1,len(ranks)+1)/ranks)))
    return {"test_images":len(kept),"queries_with_relevant_test_images":len(aps),
            "recall_at_1":float(np.mean(recalls[1])) if aps else None,
            "recall_at_5":float(np.mean(recalls[5])) if aps else None,
            "recall_at_10":float(np.mean(recalls[10])) if aps else None,
            "mean_average_precision":float(np.mean(aps)) if aps else None}

def latency_ms(model,device):
    x=torch.zeros((1,3,224,224),device=device); model.eval()
    with torch.inference_mode():
        for _ in range(3): model.embed(x)
        if device.type=="cuda": torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(20): model.embed(x)
        if device.type=="cuda": torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/20

def evaluate_model(name,kind,path,device):
    state=torch.load(path,map_location=device,weights_only=False)
    run_info_path=Path("artifacts/metrics")/f"{path.stem}_run.json"
    run_info=json.loads(run_info_path.read_text(encoding="utf-8")) if run_info_path.exists() else {}
    cls=SiameseNetwork if kind=="siamese" else TripletNetwork
    model=cls(int(state["embedding_dim"]),pretrained=False).to(device); model.load_state_dict(state["model_state_dict"]); model.eval()
    yv,sv=pair_scores(model,"data/splits/validation.csv",device); threshold=select_threshold(yv,sv)
    yt,st=pair_scores(model,"data/splits/test.csv",device); pred=st>=threshold; cm=confusion_matrix(yt,pred,labels=[0,1])
    fpr,tpr,_=roc_curve(yt,st); precision,recall,_=precision_recall_curve(yt,st)
    calibration={"threshold":float(threshold),"validation_precision":float(precision_score(yv,sv>=threshold,zero_division=0)),
        "validation_recall":float(recall_score(yv,sv>=threshold,zero_division=0)),"validation_f1":float(f1_score(yv,sv>=threshold,zero_division=0)),
        "selection_method":"maximize validation F1 over observed validation cosine similarities; ties prefer precision then higher threshold",
        "calibrated_at_utc":datetime.now(timezone.utc).isoformat(),"checkpoint_sha256":sha256_file(path),"validation_pairs":int(len(yv))}
    metrics={"model":name,"checkpoint":str(path),"checkpoint_sha256":calibration["checkpoint_sha256"],"device":str(device),
        "epochs_completed":int(run_info.get("epochs_completed",state.get("epoch",-1)+1)),
        "selected_checkpoint_epoch":int(state.get("epoch",-1)+1),"training_runtime":run_info,
        "training_config":state.get("config",{}),
        "embedding_dimension":int(state["embedding_dim"]),"calibration":calibration,
        "test_pairs":int(len(yt)),"test_positive_pairs":int(yt.sum()),"test_negative_pairs":int((yt==0).sum()),
        "accuracy":float(accuracy_score(yt,pred)),"precision":float(precision_score(yt,pred,zero_division=0)),
        "recall":float(recall_score(yt,pred,zero_division=0)),"f1":float(f1_score(yt,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(yt,st)) if len(np.unique(yt))>1 else None,
        "pr_auc":float(average_precision_score(yt,st)),"average_precision":float(average_precision_score(yt,st)),
        "confusion_matrix":{"tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])},
        "retrieval":retrieval_metrics(model,"data/splits/test.csv",device),"inference_latency_ms_per_image":latency_ms(model,device),"checkpoint_size_bytes":Path(path).stat().st_size}
    return metrics,(fpr,tpr,precision,recall)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs=[("Siamese ResNet50 (visual negatives)", "siamese", Path("artifacts/checkpoints/siamese_resnet50.pt")),
             ("Siamese ResNet50 (random negatives)", "siamese", Path("artifacts/checkpoints/siamese_resnet50_random.pt")),
             ("Triplet ResNet50 (batch-hard)", "triplet", Path("artifacts/checkpoints/triplet_resnet50.pt")),
             ("Triplet ResNet50 (random negatives)", "triplet", Path("artifacts/checkpoints/triplet_resnet50_random.pt"))]
    results=[]; curves={}
    for name,kind,path in configs:
        if path.exists():
            metrics,curve=evaluate_model(name,kind,path,device); results.append(metrics); curves[name]=curve
            if path.name=="siamese_resnet50.pt":
                Path("artifacts/metrics/calibration.json").write_text(json.dumps(metrics["calibration"],indent=2),encoding="utf-8")
    if not results: raise RuntimeError("No trained model checkpoints were found.")
    out=Path("artifacts/metrics"); out.mkdir(parents=True,exist_ok=True)
    selected=max(results,key=lambda r:(r["calibration"]["validation_f1"],r["calibration"]["validation_precision"]))
    selection={"model":selected["model"],"checkpoint":selected["checkpoint"],
        "checkpoint_sha256":selected["checkpoint_sha256"],"calibration":selected["calibration"],
        "selection_method":"highest validation F1 among the evaluated Siamese and Triplet ResNet50 runs; ties prefer validation precision. DINOv2 is retrieval-only and not threshold-classified.",
        "selected_at_utc":datetime.now(timezone.utc).isoformat()}
    (out/"selected_model.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    evaluation={"dataset_scope":"Refined ABO handbags / purses / fashion totes; identity-separated splits",
        "threshold_policy":"validation-only F1 calibration per checkpoint; no threshold selection on test data",
        "device":str(device),"models":results,"model_comparisons":results,"selected_model":selection,
        "calibration":selection["calibration"]}
    dino_path=out/"dinov2_retrieval.json"
    if dino_path.exists(): evaluation["frozen_dinov2_retrieval"]=json.loads(dino_path.read_text(encoding="utf-8"))
    (out/"evaluation.json").write_text(json.dumps(evaluation,indent=2),encoding="utf-8")
    with (out/"model_comparison.csv").open("w",encoding="utf-8",newline="") as f:
        cols=["model","epochs_completed","accuracy","precision","recall","f1","roc_auc","pr_auc","recall_at_1","recall_at_5","recall_at_10","mean_average_precision","inference_latency_ms_per_image","checkpoint_size_bytes"]
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in results:
            row={key:r.get(key) for key in cols if key in r}
            row.update({f"recall_at_{k}":r["retrieval"][f"recall_at_{k}"] for k in (1,5,10)})
            row["mean_average_precision"]=r["retrieval"]["mean_average_precision"]
            w.writerow(row)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for r in results:
        fpr,tpr,_,_=curves[r["model"]]; ax.plot(fpr,tpr,label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],"--",color="#aaa");ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Identity-held-out test ROC");ax.legend();fig.tight_layout();fig.savefig(out/"roc.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots()
    for r in results:
        _,_,precision,recall=curves[r["model"]]; ax.plot(recall,precision,label=f"{r['model']} AP={r['pr_auc']:.3f}")
    ax.set(xlabel="Recall",ylabel="Precision",title="Identity-held-out test precision–recall");ax.legend();fig.tight_layout();fig.savefig(out/"precision_recall.png",dpi=160);plt.close(fig)
    print(json.dumps(evaluation,indent=2))
if __name__=="__main__": main()
