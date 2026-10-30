import csv, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from ml.datasets.pairs import PairDataset
from ml.models.siamese import SiameseNetwork, TripletNetwork

def pair_scores(model, manifest, device):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset=PairDataset(manifest,transform)
    loader=DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0)
    labels=[];scores=[];model.eval()
    with torch.inference_mode():
        for a,b,y in loader:
            ea,eb=model.embed(a.to(device)),model.embed(b.to(device)); scores.extend((ea*eb).sum(1).cpu().numpy().tolist());labels.extend(y.numpy().tolist())
    return np.asarray(labels),np.asarray(scores)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");checkpoint=torch.load("artifacts/checkpoints/siamese_resnet50.pt",map_location=device,weights_only=False)
    model=SiameseNetwork(int(checkpoint["embedding_dim"]),pretrained=False).to(device);model.load_state_dict(checkpoint["model_state_dict"])
    yv,sv=pair_scores(model,"data/splits/validation.csv",device)
    thresholds=np.unique(np.concatenate([sv,[0.0,1.0]]));best=max(thresholds,key=lambda t:f1_score(yv,sv>=t,zero_division=0))
    yt,st=pair_scores(model,"data/splits/test.csv",device);pred=st>=best
    cm=confusion_matrix(yt,pred,labels=[0,1]);fpr,tpr,_=roc_curve(yt,st);precision,recall,_=precision_recall_curve(yt,st)
    metrics={"model":"Siamese ResNet50","checkpoint":"artifacts/checkpoints/siamese_resnet50.pt","device":str(device),"embedding_dimension":int(checkpoint["embedding_dim"]),"threshold_selected_on_validation":float(best),"validation_pairs":int(len(yv)),"test_pairs":int(len(yt)),"test_positive_pairs":int(yt.sum()),"test_negative_pairs":int((yt==0).sum()),"accuracy":float(accuracy_score(yt,pred)),"precision":float(precision_score(yt,pred,zero_division=0)),"recall":float(recall_score(yt,pred,zero_division=0)),"f1":float(f1_score(yt,pred,zero_division=0)),"roc_auc":float(roc_auc_score(yt,st)) if len(np.unique(yt))>1 else None,"average_precision":float(average_precision_score(yt,st)),"confusion_matrix":{"tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])}}
    comparisons=[dict(metrics)]
    triplet_path=Path("artifacts/checkpoints/triplet_resnet50.pt")
    if triplet_path.exists():
        ts=torch.load(triplet_path,map_location=device,weights_only=False);triplet=TripletNetwork(int(ts["embedding_dim"]),pretrained=False).to(device);triplet.load_state_dict(ts["model_state_dict"])
        tv,psv=pair_scores(triplet,"data/splits/validation.csv",device);tt,pst=pair_scores(triplet,"data/splits/test.csv",device)
        trip_threshold=max(np.unique(np.concatenate([psv,[0.0,1.0]])),key=lambda t:f1_score(tv,psv>=t,zero_division=0));trip_pred=pst>=trip_threshold;trip_cm=confusion_matrix(tt,trip_pred,labels=[0,1])
        trip_metrics={"model":"Triplet ResNet50","checkpoint":str(triplet_path),"device":str(device),"embedding_dimension":int(ts["embedding_dim"]),"threshold_selected_on_validation":float(trip_threshold),"validation_pairs":int(len(tv)),"test_pairs":int(len(tt)),"test_positive_pairs":int(tt.sum()),"test_negative_pairs":int((tt==0).sum()),"accuracy":float(accuracy_score(tt,trip_pred)),"precision":float(precision_score(tt,trip_pred,zero_division=0)),"recall":float(recall_score(tt,trip_pred,zero_division=0)),"f1":float(f1_score(tt,trip_pred,zero_division=0)),"roc_auc":float(roc_auc_score(tt,pst)) if len(np.unique(tt))>1 else None,"average_precision":float(average_precision_score(tt,pst)),"confusion_matrix":{"tn":int(trip_cm[0,0]),"fp":int(trip_cm[0,1]),"fn":int(trip_cm[1,0]),"tp":int(trip_cm[1,1])}}
        comparisons.append(trip_metrics);Path("artifacts/metrics/triplet_evaluation.json").write_text(json.dumps(trip_metrics,indent=2),encoding="utf-8")
    metrics["model_comparisons"]=comparisons
    out=Path("artifacts/metrics");out.mkdir(parents=True,exist_ok=True);(out/"evaluation.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with (out/"confusion_matrix.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f);w.writerow(["actual/predicted","different","same"]);w.writerow(["different",cm[0,0],cm[0,1]]);w.writerow(["same",cm[1,0],cm[1,1]])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot(fpr,tpr,label=f"Siamese (AUC={metrics['roc_auc']:.3f})");ax.plot([0,1],[0,1],"--",color="#aaa");ax.set(xlabel="False positive rate",ylabel="True positive rate",title="Held-out identity test ROC");ax.legend();fig.tight_layout();fig.savefig(out/"roc.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots();ax.plot(recall,precision,label=f"AP={metrics['average_precision']:.3f}");ax.set(xlabel="Recall",ylabel="Precision",title="Held-out identity test precision–recall");ax.legend();fig.tight_layout();fig.savefig(out/"precision_recall.png",dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
