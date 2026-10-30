from collections import defaultdict
from pathlib import Path
import csv, random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """Product-identity pairs with optional visually mined different-product negatives."""
    def __init__(self, manifest: str | Path, transform=None, seed: int = 42,
                 negative_candidates: dict[str, list[str]] | None = None, negative_mode="visual"):
        self.transform, self.rng = transform, random.Random(seed)
        with Path(manifest).open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
        identities = defaultdict(list)
        for row in rows:
            if Path(row["image_path"]).is_file(): identities[row["product_identity"]].append(row["image_path"])
        self.groups = {k: v for k, v in identities.items() if len(v) >= 2}
        self.singletons = {k: v[0] for k, v in identities.items() if len(v) == 1}
        self.identity_keys = list(self.groups)
        self.rows = []
        for key, paths in self.groups.items():
            for a in paths:
                candidates = [b for b in paths if b != a]
                self.rows.append((key, a, self.rng.choice(candidates), 1))
        all_groups = {**self.groups, **{k: [v] for k, v in self.singletons.items()}}
        all_keys = list(all_groups)
        if len(all_keys) >= 2:
            by_key = {key: paths for key, paths in all_groups.items()}
            for key, paths in all_groups.items():
                eligible = [candidate for candidate in (negative_candidates or {}).get(key, [])
                            if candidate in by_key and candidate != key]
                if negative_mode == "visual" and eligible:
                    near = eligible[:min(5, len(eligible))]
                else:
                    near = [candidate for candidate in all_keys if candidate != key]
                for a in paths:
                    other = self.rng.choice(near)
                    self.rows.append((key, a, self.rng.choice(by_key[other]), 0))
        if not self.rows: raise ValueError("No labeled image pairs available. Each positive requires >=2 images for one product identity.")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        _, a, b, same = self.rows[idx]
        with Image.open(a) as im: image_a = im.convert("RGB")
        with Image.open(b) as im: image_b = im.convert("RGB")
        if self.transform: image_a, image_b = self.transform(image_a), self.transform(image_b)
        return image_a, image_b, same

class TripletDataset(Dataset):
    def __init__(self, manifest, transform=None, seed=42):
        self.pairs = PairDataset(manifest, transform, seed)
        self.transform = transform
        self.groups = self.pairs.groups
        self.keys = list(self.groups)
        self.triplets = [(k, a, b) for k, paths in self.groups.items() for a, b in zip(paths, paths[1:])]
        if len(self.keys) < 2: raise ValueError("Triplet training needs at least two identities with >=2 images each.")
        self.rng = random.Random(seed)
    def __len__(self): return len(self.triplets)
    def __getitem__(self, idx):
        key, anchor, positive = self.triplets[idx]
        negative_key = self.rng.choice([k for k in self.keys if k != key])
        paths = self.groups[negative_key]
        negative = self.rng.choice(paths)
        imgs = []
        for path in (anchor, positive, negative):
            with Image.open(path) as im: imgs.append(im.convert("RGB"))
        if self.transform: imgs = [self.transform(im) for im in imgs]
        return tuple(imgs)

class IdentityImageDataset(Dataset):
    """Image/identity pairs for online batch-hard or random triplet construction."""
    def __init__(self, manifest, transform=None):
        with Path(manifest).open(encoding="utf-8", newline="") as f: rows=list(csv.DictReader(f))
        paths=[]; labels=[]; keys=sorted({r["product_identity"] for r in rows})
        self.identity_to_label={key:i for i,key in enumerate(keys)}
        for row in rows:
            if Path(row["image_path"]).is_file():
                paths.append(row["image_path"]); labels.append(self.identity_to_label[row["product_identity"]])
        self.paths,self.labels,self.transform=paths,labels,transform
        self.groups=defaultdict(list)
        for i,label in enumerate(labels): self.groups[label].append(i)
    def __len__(self): return len(self.paths)
    def __getitem__(self,idx):
        with Image.open(self.paths[idx]) as im: image=im.convert("RGB")
        if self.transform: image=self.transform(image)
        return image,self.labels[idx]

class IdentityBatchSampler:
    """P identities × K images, with true distinct product identities per batch."""
    def __init__(self,dataset, batch_size=16, identities_per_batch=4, seed=42):
        self.groups=dataset.groups; self.keys=[key for key,items in self.groups.items() if len(items)>=2]; self.seed=seed
        self.p=max(2,min(identities_per_batch,len(self.keys)))
        self.k=min(max(2,batch_size//self.p),min((len(self.groups[key]) for key in self.keys),default=2)); self.batch_size=self.p*self.k
        self.steps=max(1,(len(dataset)+self.batch_size-1)//self.batch_size)
    def __len__(self): return self.steps
    def __iter__(self):
        rng=random.Random(self.seed)
        for _ in range(self.steps):
            selected=rng.sample(self.keys,self.p); batch=[]
            for key in selected:
                pool=self.groups[key]
                batch.extend(rng.sample(pool,self.k))
            yield batch

def validate_image(path: str | Path, min_side: int = 32):
    try:
        try:
            import cv2
            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if decoded is None: return False, 0, 0
            height, width = decoded.shape[:2]
            return width >= min_side and height >= min_side, width, height
        except ImportError:
            pass
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.width >= min_side and im.height >= min_side, im.width, im.height
    except (OSError, UnidentifiedImageError): return False, 0, 0
