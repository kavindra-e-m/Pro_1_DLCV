"""Identity-aware triplet training with random or online batch-hard negatives."""
import argparse, json, random, platform, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ml.datasets.pairs import IdentityImageDataset, IdentityBatchSampler
from ml.models.siamese import TripletNetwork

def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,"cudnn"): torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def mine_loss(embeddings, labels, margin=.3, mode="batch-hard"):
    labels=labels.reshape(-1,1)
    same=labels.eq(labels.T); eye=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    positive=same & ~eye; negative=~same
    distance=torch.cdist(embeddings,embeddings,p=2)
    valid=positive.any(1) & negative.any(1)
    if not valid.any(): return None
    hard_positive=distance.masked_fill(~positive,-1).max(1).values
    if mode=="batch-hard":
        negative_distance=distance.masked_fill(~negative,float("inf"))
        hard_negative=negative_distance.min(1).values
    else:
        # Reproducible random valid in-batch negatives, for an apples-to-apples baseline.
        choices=[]
        for i in range(len(labels)):
            candidates=torch.where(negative[i])[0]
            choices.append(candidates[torch.randint(len(candidates),(1,),device=labels.device)].item() if len(candidates) else i)
        hard_negative=distance[torch.arange(len(labels),device=labels.device),torch.tensor(choices,device=labels.device)]
    losses=torch.relu(hard_positive[valid]-hard_negative[valid]+margin)
    return losses.mean()

def run_epoch(model, loader, device, mode, optimizer=None, train_last_block=False):
    training=optimizer is not None; model.train(training); total=0.; count=0
    if training and train_last_block:
        model.backbone.encoder.eval(); model.backbone.encoder.layer4.train(); model.backbone.projection.train()
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            embeddings=model.embed(images)
            loss=mine_loss(embeddings,labels,mode=mode)
            if loss is None: continue
            if training: loss.backward(); optimizer.step()
        total+=float(loss.item())*len(labels); count+=len(labels)
    return total/max(count,1)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--identities-per-batch",type=int,default=4); p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--embedding-dim",type=int,default=256); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--patience",type=int,default=5); p.add_argument("--resume",action="store_true")
    p.add_argument("--mining",choices=("batch-hard","random"),default="batch-hard"); p.add_argument("--freeze-backbone",action="store_true");p.add_argument("--train-last-block",action="store_true"); p.add_argument("--scratch",action="store_true");p.add_argument("--run-name",default="triplet_resnet50")
    a=p.parse_args(); seed_all(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started=datetime.now(timezone.utc);started=time.perf_counter()
    train_tf=transforms.Compose([transforms.Resize((256,256)),transforms.RandomResizedCrop(224,scale=(.75,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    val_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train=IdentityImageDataset("data/splits/train.csv",train_tf); val=IdentityImageDataset("data/splits/validation.csv",val_tf)
    sampler=IdentityBatchSampler(train,a.batch_size,a.identities_per_batch,a.seed)
    val_sampler=IdentityBatchSampler(val,a.batch_size,a.identities_per_batch,a.seed)
    loader=DataLoader(train,batch_sampler=sampler,num_workers=0); valloader=DataLoader(val,batch_sampler=val_sampler,num_workers=0)
    model=TripletNetwork(a.embedding_dim,pretrained=not a.scratch).to(device)
    if a.freeze_backbone and a.train_last_block: raise ValueError("Choose either --freeze-backbone or --train-last-block.")
    if a.freeze_backbone:
        for parameter in model.backbone.encoder.parameters(): parameter.requires_grad=False
    elif a.train_last_block:
        for parameter in model.backbone.encoder.parameters(): parameter.requires_grad=False
        for parameter in model.backbone.encoder.layer4.parameters(): parameter.requires_grad=True
    opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=2,factor=.5)
    out=Path(f"artifacts/checkpoints/{a.run_name}.pt"); last=Path(f"artifacts/checkpoints/{a.run_name}_last.pt")
    out.parent.mkdir(parents=True,exist_ok=True); start=0; best=float("inf"); stale=0; history=[]
    if a.resume and last.exists():
        state=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"])
        if state.get("training_manifest_sha256") not in (None,file_hash("data/splits/train.csv")):
            raise ValueError("Training manifest changed since this checkpoint; use a new --run-name.")
        opt.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"): scheduler.load_state_dict(state["scheduler_state_dict"])
        start=state["epoch"]+1; best=state.get("best_val_loss",best)
        stale=state.get("stale_epochs",0); history=state.get("history",[])
    for epoch in range(start,a.epochs):
        sampler.seed=a.seed+epoch; val_sampler.seed=a.seed+epoch
        train_loss=run_epoch(model,loader,device,a.mining,opt,a.train_last_block)
        val_loss=run_epoch(model,valloader,device,"batch-hard")
        scheduler.step(val_loss); row={"epoch":epoch+1,"train_triplet_loss":train_loss,"validation_triplet_loss":val_loss,"learning_rate":opt.param_groups[0]["lr"]}; history.append(row)
        if val_loss<best:
            best=val_loss; stale=0
            torch.save({"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"best_val_loss":best,"history":history,"config":vars(a),"device":str(device),"training_manifest_sha256":file_hash("data/splits/train.csv")},out.with_suffix(".tmp"));out.with_suffix(".tmp").replace(out)
        else: stale+=1
        state={"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"best_val_loss":best,"stale_epochs":stale,"history":history,"config":vars(a),"device":str(device),"training_manifest_sha256":file_hash("data/splits/train.csv")}
        torch.save(state,last.with_suffix(".tmp"));last.with_suffix(".tmp").replace(last)
        metric=Path("artifacts/metrics")/("triplet_training.json" if a.run_name=="triplet_resnet50" else f"{a.run_name}_training.json");metric.parent.mkdir(parents=True,exist_ok=True);metric.write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(json.dumps({**row,"best_validation_loss":best,"device":str(device),"mining":a.mining}),flush=True)
        if stale>=a.patience: print("Early stopping."); break
    run_info={"run_name":a.run_name,"config":vars(a),"device":str(device),"hardware":platform.processor(),
        "started_at_utc":run_started.isoformat(),"finished_at_utc":datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds":time.perf_counter()-started,"epochs_completed":len(history),"seed":a.seed}
    Path("artifacts/metrics",f"{a.run_name}_run.json").write_text(json.dumps(run_info,indent=2),encoding="utf-8")

if __name__=="__main__": main()
