from collections import defaultdict
from pathlib import Path
import csv, random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """Identity-preserving pair dataset. Positive pairs require distinct images of one real identity."""
    def __init__(self, manifest: str | Path, transform=None, seed: int = 42):
        self.transform, self.rng = transform, random.Random(seed)
        with Path(manifest).open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
        identities = defaultdict(list)
        for row in rows:
            if Path(row["image_path"]).is_file(): identities[row["product_identity"]].append(row["image_path"])
        self.groups = {k: v for k, v in identities.items() if len(v) >= 2}
        self.singletons = {k: v[0] for k, v in identities.items() if len(v) == 1}
        self.identity_keys = list(self.groups)
        self.rows = [(key, a, b, 1) for key, paths in self.groups.items() for a, b in zip(paths, paths[1:])]
        all_groups = {**self.groups, **{k: [v] for k, v in self.singletons.items()}}
        all_keys = list(all_groups)
        if len(all_keys) >= 2:
            for key, paths in all_groups.items():
                other = self.rng.choice([k for k in all_keys if k != key])
                for a in paths: self.rows.append((key, a, self.rng.choice(all_groups[other]), 0))
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
