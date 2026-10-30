import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ml.datasets.pairs import TripletDataset
from ml.models.siamese import TripletNetwork

def main():
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=15);p.add_argument("--batch-size",type=int,default=12);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--embedding-dim",type=int,default=256);p.add_argument("--seed",type=int,default=42);p.add_argument("--patience",type=int,default=4);p.add_argument("--resume",action="store_true");p.add_argument("--freeze-backbone",action="store_true");p.add_argument("--scratch",action="store_true");a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf=transforms.Compose([transforms.Resize((256,256)),transforms.RandomResizedCrop(224,scale=(.75,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    val_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset=TripletDataset("data/splits/train.csv",train_tf,a.seed);loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,num_workers=0)
    valset=TripletDataset("data/splits/validation.csv",val_tf,a.seed);valloader=DataLoader(valset,batch_size=a.batch_size,shuffle=False,num_workers=0)
    model=TripletNetwork(a.embedding_dim,pretrained=not a.scratch).to(device)
    if a.freeze_backbone:
        for parameter in model.backbone.encoder.parameters(): parameter.requires_grad=False
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=2,factor=.5);criterion=torch.nn.TripletMarginLoss(margin=.3,p=2);history=[]
    out=Path("artifacts/checkpoints/triplet_resnet50.pt");last=Path("artifacts/checkpoints/triplet_resnet50_last.pt");out.parent.mkdir(parents=True,exist_ok=True);start=0;best=float("inf");stale=0
    if a.resume and last.exists():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state["model_state_dict"]);opt.load_state_dict(state["optimizer_state_dict"]);start=state["epoch"]+1;best=state.get("best_val_loss",best);stale=state.get("stale_epochs",0);history=state.get("history",[])
    for epoch in range(start,a.epochs):
        model.train()
        if a.freeze_backbone: model.backbone.encoder.eval()
        total=0;count=0
        for anchor,positive,negative in loader:
            anchor,positive,negative=anchor.to(device),positive.to(device),negative.to(device);opt.zero_grad(set_to_none=True);ea,ep,en=model(anchor,positive,negative);loss=criterion(ea,ep,en);loss.backward();opt.step();total+=loss.item()*len(anchor);count+=len(anchor)
        train_loss=total/max(count,1);model.eval();vtotal=0;vcount=0
        with torch.inference_mode():
            for anchor,positive,negative in valloader:
                ea,ep,en=model(anchor.to(device),positive.to(device),negative.to(device));loss=criterion(ea,ep,en);vtotal+=loss.item()*len(anchor);vcount+=len(anchor)
        val_loss=vtotal/max(vcount,1);scheduler.step(val_loss);row={"epoch":epoch+1,"train_triplet_loss":train_loss,"validation_triplet_loss":val_loss,"learning_rate":opt.param_groups[0]["lr"]};history.append(row)
        if val_loss<best:
            best=val_loss;stale=0;torch.save({"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"best_val_loss":best,"history":history,"config":vars(a)},out.with_suffix(".tmp"));out.with_suffix(".tmp").replace(out)
        else:stale+=1
        state={"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"best_val_loss":best,"stale_epochs":stale,"history":history,"config":vars(a)};torch.save(state,last.with_suffix(".tmp"));last.with_suffix(".tmp").replace(last)
        print(json.dumps(row),flush=True)
        if stale>=a.patience: print("Early stopping.");break
    Path("artifacts/metrics/triplet_training.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
if __name__=="__main__":main()
