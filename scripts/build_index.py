import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ml.models.siamese import SiameseNetwork

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--batch-size",type=int,default=16);args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");state=torch.load("artifacts/checkpoints/siamese_resnet50.pt",map_location=device,weights_only=False)
    model=SiameseNetwork(int(state["embedding_dim"]),pretrained=False).to(device);model.load_state_dict(state["model_state_dict"]);model.eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with open("data/processed/metadata/catalog.csv",encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    vectors=[];kept=[]
    with torch.inference_mode():
        for start in range(0,len(rows),max(1,args.batch_size)):
            tensors=[];batch_rows=[]
            for row in rows[start:start+max(1,args.batch_size)]:
                try:
                    with Image.open(row["image_path"]) as image: tensors.append(tf(image.convert("RGB")))
                    batch_rows.append(row)
                except (OSError,ValueError) as exc: print(f"skip {row['image_path']}: {exc}")
            if tensors:
                features=model.embed(torch.stack(tensors).to(device)).cpu().numpy()
                vectors.extend(features);kept.extend(batch_rows)
    if not vectors: raise RuntimeError("No valid catalog images available for indexing.")
    matrix=np.asarray(vectors,dtype="float32");matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12)
    out=Path("artifacts/embeddings");out.mkdir(parents=True,exist_ok=True);np.save(out/"catalog.npy",matrix)
    dest=Path("artifacts/faiss_index");dest.mkdir(parents=True,exist_ok=True);index_path=dest/"catalog.index"
    backend="numpy_exact"
    try:
        import faiss
        index=faiss.IndexFlatIP(matrix.shape[1]);index.add(matrix);faiss.write_index(index,str(index_path));backend="faiss.IndexFlatIP"
    except ImportError:
        index_path.write_text(json.dumps({"backend":backend,"vectors":"../embeddings/catalog.npy"}),encoding="utf-8")
    with (out/"catalog.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=kept[0].keys());writer.writeheader();writer.writerows(kept)
    (dest/"index_metadata.json").write_text(json.dumps({"count":len(kept),"dimension":int(matrix.shape[1]),"backend":backend,"checkpoint":"artifacts/checkpoints/siamese_resnet50.pt"},indent=2),encoding="utf-8")
    print(f"Indexed {len(kept)} genuine images with {backend}.")
if __name__=="__main__":main()
