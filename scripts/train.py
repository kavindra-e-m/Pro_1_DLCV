import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ml.datasets.pairs import PairDataset
from ml.models.siamese import SiameseNetwork, ContrastiveLoss

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, criterion, device, optimizer=None):
    training=optimizer is not None; model.train(training); total=0.; count=0
    for a,b,same in loader:
        a,b,same=a.to(device),b.to(device),same.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            ea,eb=model(a,b); loss=criterion(ea,eb,same)
            if training: loss.backward(); optimizer.step()
        total += loss.item()*len(same); count += len(same)
    return total/max(count,1)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--embedding-dim",type=int,default=256);p.add_argument("--patience",type=int,default=5);p.add_argument("--seed",type=int,default=42);p.add_argument("--resume",action="store_true");p.add_argument("--scratch",action="store_true");a=p.parse_args()
    seed_all(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.RandomResizedCrop(224,scale=(.75,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train=PairDataset("data/splits/train.csv",transform,a.seed); val=PairDataset("data/splits/validation.csv",eval_transform,a.seed)
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
    model=SiameseNetwork(a.embedding_dim,pretrained=not a.scratch).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=2,factor=.5); loss_fn=ContrastiveLoss()
    checkpoint=Path("artifacts/checkpoints/siamese_resnet50.pt"); last_checkpoint=Path("artifacts/checkpoints/siamese_resnet50_last.pt"); checkpoint.parent.mkdir(parents=True,exist_ok=True); start=0; best=float("inf"); stale=0; history=[]
    if a.resume and last_checkpoint.exists():
        state=torch.load(last_checkpoint,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); start=state["epoch"]+1; best=state.get("best_val_loss",best); stale=state.get("stale_epochs",0); history=state.get("history",[])
    for epoch in range(start,a.epochs):
        train_loss=run_epoch(model,train_loader,loss_fn,device,optimizer); val_loss=run_epoch(model,val_loader,loss_fn,device); scheduler.step(val_loss)
        history.append({"epoch":epoch+1,"train_loss":train_loss,"validation_loss":val_loss,"learning_rate":optimizer.param_groups[0]["lr"]})
        if val_loss < best:
            best=val_loss;stale=0
            torch.save({"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"best_val_loss":best,"history":history,"config":vars(a),"seed":a.seed,"device":str(device)},checkpoint.with_suffix(".tmp"));checkpoint.with_suffix(".tmp").replace(checkpoint)
        else: stale+=1
        state={"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"best_val_loss":best,"stale_epochs":stale,"history":history,"config":vars(a),"seed":a.seed,"device":str(device)}
        torch.save(state,last_checkpoint.with_suffix(".tmp"));last_checkpoint.with_suffix(".tmp").replace(last_checkpoint)
        Path("artifacts/metrics/training.json").parent.mkdir(parents=True,exist_ok=True);Path("artifacts/metrics/training.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(f"epoch {epoch+1}/{a.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f} device={device}",flush=True)
        if stale>=a.patience: print("Early stopping.");break
if __name__=="__main__":main()
