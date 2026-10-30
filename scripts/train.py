import argparse, csv, json, random, platform, time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ml.datasets.pairs import PairDataset
from ml.models.siamese import SiameseNetwork, ContrastiveLoss

def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,"cudnn"): torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def run_epoch(model, loader, criterion, device, optimizer=None, train_last_block=False):
    training=optimizer is not None; model.train(training); total=0.; count=0
    if training and train_last_block:
        model.backbone.encoder.eval()
        model.backbone.encoder.layer4.train()
        model.backbone.projection.train()
    for a,b,same in loader:
        a,b,same=a.to(device),b.to(device),same.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            ea,eb=model(a,b); loss=criterion(ea,eb,same)
            if training: loss.backward(); optimizer.step()
        total += loss.item()*len(same); count += len(same)
    return total/max(count,1)

def mine_visual_neighbors(model, manifest, transform, device, batch_size=8):
    """Mine different ABO product IDs that are nearest in pretrained ResNet features."""
    from PIL import Image
    with Path(manifest).open(encoding="utf-8", newline="") as f: rows=list(csv.DictReader(f))
    model.eval(); vectors=[]; identities=[]
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images=[]; keys=[]
            for row in rows[start:start+batch_size]:
                try:
                    with Image.open(row["image_path"]) as im: images.append(transform(im.convert("RGB")))
                    keys.append(row["product_identity"])
                except (OSError, ValueError): continue
            if images:
                feature=model.backbone.encoder(torch.stack(images).to(device))
                vectors.extend(torch.nn.functional.normalize(feature,dim=1).cpu().numpy())
                identities.extend(keys)
    means={}
    for key in sorted(set(identities)):
        indices=[i for i,k in enumerate(identities) if k==key]
        mean=np.mean(np.asarray([vectors[i] for i in indices]),axis=0)
        means[key]=mean/(np.linalg.norm(mean)+1e-12)
    keys=list(means); matrix=np.asarray([means[k] for k in keys]); similarity=matrix@matrix.T
    return {key:[keys[i] for i in np.argsort(-similarity[row]) if keys[i]!=key]
            for row,key in enumerate(keys)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--embedding-dim",type=int,default=256);p.add_argument("--patience",type=int,default=5);p.add_argument("--seed",type=int,default=42);p.add_argument("--resume",action="store_true");p.add_argument("--scratch",action="store_true");p.add_argument("--negative-mode",choices=("visual","random"),default="visual");p.add_argument("--run-name",default="siamese_resnet50");p.add_argument("--train-last-block",action="store_true");a=p.parse_args()
    seed_all(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_started=datetime.now(timezone.utc); started=time.perf_counter()
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.RandomResizedCrop(224,scale=(.75,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    model=SiameseNetwork(a.embedding_dim,pretrained=not a.scratch).to(device)
    if a.train_last_block:
        for parameter in model.backbone.encoder.parameters(): parameter.requires_grad=False
        for parameter in model.backbone.encoder.layer4.parameters(): parameter.requires_grad=True
    neighbors=mine_visual_neighbors(model,"data/splits/train.csv",eval_transform,device) if a.negative_mode=="visual" else {}
    train=PairDataset("data/splits/train.csv",transform,a.seed,neighbors,a.negative_mode); val=PairDataset("data/splits/validation.csv",eval_transform,a.seed,negative_mode="random")
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=2,factor=.5); loss_fn=ContrastiveLoss()
    checkpoint=Path(f"artifacts/checkpoints/{a.run_name}.pt"); last_checkpoint=Path(f"artifacts/checkpoints/{a.run_name}_last.pt"); checkpoint.parent.mkdir(parents=True,exist_ok=True); start=0; best=float("inf"); stale=0; history=[]
    if a.resume and last_checkpoint.exists():
        state=torch.load(last_checkpoint,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("training_manifest_sha256") not in (None,file_hash("data/splits/train.csv")):
            raise ValueError("Training manifest changed since this checkpoint; start a new run name instead of resuming.")
        if state.get("scheduler_state_dict"): scheduler.load_state_dict(state["scheduler_state_dict"])
        start=state["epoch"]+1; best=state.get("best_val_loss",best); stale=state.get("stale_epochs",0); history=state.get("history",[])
        prior_batch_size=state.get("config",{}).get("batch_size")
        for item in history:
            item.setdefault("batch_size",prior_batch_size)
    for epoch in range(start,a.epochs):
        train_loss=run_epoch(model,train_loader,loss_fn,device,optimizer,a.train_last_block); val_loss=run_epoch(model,val_loader,loss_fn,device); scheduler.step(val_loss)
        history.append({"epoch":epoch+1,"train_loss":train_loss,"validation_loss":val_loss,"learning_rate":optimizer.param_groups[0]["lr"],"batch_size":a.batch_size,"train_last_block":a.train_last_block,"negative_mode":a.negative_mode})
        if val_loss < best:
            best=val_loss;stale=0
            torch.save({"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"best_val_loss":best,"history":history,"config":vars(a),"seed":a.seed,"device":str(device),"training_manifest_sha256":file_hash("data/splits/train.csv"),"negative_mining":"pretrained_resnet50_visual_nearest_different_product" if a.negative_mode=="visual" else "random_different_product"},checkpoint.with_suffix(".tmp"));checkpoint.with_suffix(".tmp").replace(checkpoint)
        else: stale+=1
        state={"epoch":epoch,"embedding_dim":a.embedding_dim,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"best_val_loss":best,"stale_epochs":stale,"history":history,"config":vars(a),"seed":a.seed,"device":str(device),"training_manifest_sha256":file_hash("data/splits/train.csv")}
        torch.save(state,last_checkpoint.with_suffix(".tmp"));last_checkpoint.with_suffix(".tmp").replace(last_checkpoint)
        metric_path=Path("artifacts/metrics")/("training.json" if a.run_name=="siamese_resnet50" else f"{a.run_name}_training.json")
        metric_path.parent.mkdir(parents=True,exist_ok=True);metric_path.write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(f"epoch {epoch+1}/{a.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f} device={device}",flush=True)
        if stale>=a.patience: print("Early stopping.");break
    run_info={"run_name":a.run_name,"config":vars(a),"device":str(device),"hardware":platform.processor(),
        "started_at_utc":run_started.isoformat(),"finished_at_utc":datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds":time.perf_counter()-started,"epochs_completed":len(history),"seed":a.seed}
    Path("artifacts/metrics",f"{a.run_name}_run.json").write_text(json.dumps(run_info,indent=2),encoding="utf-8")
if __name__=="__main__":main()
