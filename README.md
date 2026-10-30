# BagVision AI

**BagVision AI** is a product-level visual retrieval and duplicate review application for handbag and bag catalog images. It pairs an identity-separated dataset pipeline with a PyTorch Siamese ResNet50, a triplet-learning alternative, nearest-neighbor search, and a FastAPI + React interface.

## Project status

This repository contains the end-to-end implementation and acquisition/training commands. The first real ABO handbag subset is acquired locally during setup; images, generated reports, checkpoints and indexes are intentionally excluded from Git. Evaluation scores are displayed only after they have been calculated and saved by the evaluation command. There is no pretrained project checkpoint or evaluation result committed here.

## Abstract

BagVision AI explores whether two catalog photographs represent the same product design despite changes in view, crop, background or lighting. The system learns L2-normalized image embeddings with shared ResNet50 weights and contrastive loss, evaluates on identities held out from training, and ranks catalog images by cosine similarity. A triplet-loss implementation provides an experimental alternative. The interface supports visual retrieval, pair comparison, batch duplicate candidates, catalog browsing, model metrics and dataset provenance.

## Problem and objectives

The identity target is the same **catalog product/model**, not the same physical unit. The engineering goals are to preserve genuine product identifiers and source provenance, avoid product identity leakage across splits, train real metric-learning models, measure performance on held-out identities, and expose actual inference through an application.

The system does not claim to identify whether two photos show the same physical handbag, and similarity thresholds require calibration on a relevant validation set.

## Existing research and proposed system

Research on fine-grained handbag recognition includes Wang, Li and Kot's *DeepBag: Recognizing Handbag Models* (IEEE Transactions on Multimedia, 2015). The paper reports a curated dataset of 5,545 images across 220 models. This project has not found an authorized, publicly downloadable copy of those original images and annotations; it does not claim to use DeepBag.

The proposed implementation uses genuine ABO product listings and catalog images as a source for product identities. It implements a ResNet50 embedding baseline, Siamese contrastive training, triplet training, an optional DINOv2 feature extraction baseline, and an API/UI for retrieval and duplicate review. These are engineering implementations of established methods, not a claim of research novelty.

## Data sources and licensing

| Source | Use | License / notes |
|---|---|---|
| [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html) | Primary source. Official product listing metadata is filtered for bag categories and genuine listed images are retrieved from the official small-image paths. Amazon IDs and source URLs are retained. | CC BY 4.0. Attribute Amazon.com and the dataset authors; changes are noted in the dataset report and documentation. |
| [GLAMI Entity Matching](https://huggingface.co/datasets/zidcenek/GLAMI-Entity-Matching-Dataset) | Documented optional follow-on source. Its dataset card reports ~1.3M images and multilingual product metadata. This project does not download the full 17.8 GB image release by default. | Apache 2.0 per the dataset card. The card leaves label-generation details as TODO, so its `label` field must be inspected before treating it as a product identity. |
| [DeepBag paper](https://www.ntu.edu.sg/docs/librariesprovider106/publications/object-recognition-retrieval/deepbag-recognizing-handbag-models.pdf) | Literature reference only. | Original dataset access was not confirmed; no images are represented as DeepBag. |
| [Stanford Online Products](https://www.tensorflow.org/datasets/catalog/stanford_online_products) | Optional general product retrieval benchmark, if added later. | Not handbag-specific; never report it as a handbag dataset. |
| [DINOv2 base](https://huggingface.co/facebook/dinov2-base) | Optional frozen feature encoder for an additional retrieval baseline. | Apache 2.0 per the official model card and DINOv2 repository. |

The ABO release has 147,702 product listings and 398,212 unique catalog images in its full release. Bag category detection is based on the actual product type, item name, keywords and category path. The preparation command supports an optional listing cap for a manageable run; local counts and filtering outcomes are written to `data/reports/`. The generated CSV is the source of truth for counts.

ABO attribution: Amazon.com; Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Arnab Dhua, Xi (Brian) Zhang, Tomas Yago-Vicente, Jasmine Collins, Shubham Goel and Jitendra Malik. See the [official dataset page](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html) and [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Architecture

```text
Official ABO listings + image metadata
             │ category filter and genuine image download
             ▼
Manifest → validation/hash cleanup → identity-grouped train/val/test splits
             │
             ├── Siamese ResNet50 + contrastive loss
             ├── Triplet ResNet50 + triplet margin loss
             └── DINOv2 pretrained feature extraction (optional)
                       │
       Evaluation → embeddings → exact cosine / optional FAISS index
                       │
           FastAPI + SQLite ─── React/TypeScript UI
```

See [architecture](docs/architecture.md), [methodology](docs/model_methodology.md) and [API notes](docs/api_documentation.md).

## Installation

Python 3.11 is a known supported local setup. Node.js 20+ and npm are needed for the web UI.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

For an NVIDIA GPU, install the PyTorch build matching your installed CUDA version using the [official PyTorch install selector](https://pytorch.org/get-started/locally/). CPU-only environments are supported. This project does not assume CUDA.

## End-to-end workflow

1. Acquire real ABO product records and a bounded sample of their listed images, validate them, create reports, split by product identity, and import the catalog into SQLite:

   ```powershell
   .venv\Scripts\python.exe -m scripts.prepare_data --max-products 500 --max-images-per-product 5
   ```

   Remove `--max-products` to process all matching listings. The pipeline stores raw source files and normalized images under ignored `data/` directories. See [dataset preparation](docs/dataset_documentation.md).

2. Train the Siamese metric model. The first run downloads official ImageNet pretrained ResNet50 weights unless `--scratch` is supplied. CPU training can be slow; checkpoints support resumption.

   ```powershell
   .venv\Scripts\python.exe -m scripts.train --epochs 20 --batch-size 8 --resume
   ```

3. Evaluate the best checkpoint on held-out product identities, select a similarity decision threshold on validation pairs, and save metrics/curves:

   ```powershell
   .venv\Scripts\python.exe -m scripts.evaluate
   ```

4. Build local image embeddings and the optional FAISS inner-product index:

   ```powershell
   .venv\Scripts\python.exe -m scripts.build_index
   ```

5. Start the API and frontend in separate terminals:

   ```powershell
   .venv\Scripts\python.exe -m uvicorn backend.main:app --reload
   cd frontend
   npm install
   npm run dev
   ```

6. Optional triplet experiment:

   ```powershell
   .venv\Scripts\python.exe -m scripts.train_triplet --epochs 15 --batch-size 8
   ```

7. Optional frozen DINOv2-base retrieval baseline (requires the additional Transformers dependency and downloads the public 346 MB checkpoint on first run):

   ```powershell
   uv pip install --python .venv\Scripts\python.exe -r requirements-modern.txt
   .venv\Scripts\python.exe -m scripts.extract_dinov2 --batch-size 8
   ```

The web app defaults to `http://localhost:8000/api` and `http://localhost:8000/media/abo`. Set `VITE_API_URL` and `VITE_MEDIA_URL` in `frontend/.env.local` to override them. Backend options are in `.env.example`.

## Application pages

- Overview: live SQLite counts, index/model availability and saved training history.
- Visual search: upload one image for ranked embedding neighbors.
- Compare images: cosine score and calibrated validation threshold.
- Duplicate check: candidate groups from pairwise scores for 2–100 uploads.
- Product catalog: source IDs and genuine product images.
- Model performance: saved loss history and measured held-out metrics.
- Dataset: provenance, source links, licenses and generated statistics.

No placeholder product rows or predictions are inserted. When a checkpoint or index is missing, the relevant API returns an actionable error or an empty state.

## Evaluation and current experimental results

Initial experiments were completed locally on a capped ABO subset. The command selected 250 bag listings and up to four images per listing; after decoding and exact cross-identity duplicate review, 855 images across 235 identities remained. Seed 42 produced 188 train, 23 validation and 24 test identities. A one-epoch Siamese ResNet50 fine-tuning run started from official ImageNet weights and ran on CPU (training loss 0.51474; validation loss 0.30433). The validation threshold was 0.33107. On 152 held-out test pairs (64 positive, 88 negative), observed accuracy was 69.74%, precision 61.54%, recall 75.00%, F1 0.6761, ROC AUC 0.7711 and average precision 0.6854; confusion counts were TN=58, FP=30, FN=16, TP=48. An initial one-epoch Triplet ResNet50 head run (frozen ImageNet backbone) scored accuracy 77.63%, F1 0.7606 and ROC AUC 0.8686 on the same test pairs with its own validation threshold. Frozen DINOv2-base retrieval on the 88 test images yielded recall@1 0.3256, recall@5 0.6744, recall@10 0.8488 and mAP 0.2878 (86 queries had another same-identity test image). These are preliminary results from a small, catalog-photo sample, not a production performance claim. Full local artifacts are ignored by Git; details are recorded in [experimental results](docs/experimental_results.md).

Metrics include accuracy, precision, recall, F1, ROC AUC, average precision and a confusion matrix. The threshold is selected on validation identities; the test identities remain held out. Pair metrics do not establish performance on unrelated real-world photos or physical-unit identity.

## Limitations and future work

- The first ABO images are catalog-style product photography and may not represent user photographs in the wild.
- Similarity threshold needs dataset-specific calibration and manual review for high-impact catalog merges.
- The optional GLAMI release is large; its label semantics should be verified before using labels as product identities.
- No handbag detector or cropper is claimed. Inputs are embedded as whole images.
- Performance and demographic/brand coverage depend on the actually acquired category subset.
- Future work: add documented GLAMI pair/group ingestion, cross-domain evaluation, semi-hard triplet mining, crop robustness, ONNX export, and a managed review workflow.

## References

1. J. Collins et al., “ABO: Dataset and Benchmarks for Real-World 3D Object Understanding,” CVPR 2022. [Official data page](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html).
2. Y. Wang, S. Li, and A. C. Kot, “DeepBag: Recognizing Handbag Models,” *IEEE Transactions on Multimedia*, vol. 17, no. 11, 2015. [Paper](https://www.ntu.edu.sg/docs/librariesprovider106/publications/object-recognition-retrieval/deepbag-recognizing-handbag-models.pdf).
3. R. Hadsell, S. Chopra, and Y. LeCun, “Dimensionality Reduction by Learning an Invariant Mapping,” CVPR 2006.
4. A. Hermans, L. Beyer, and B. Leibe, “In Defense of the Triplet Loss for Person Re-Identification,” arXiv:1703.07737, 2017.
5. GLAMI Entity Matching dataset card. [Hugging Face](https://huggingface.co/datasets/zidcenek/GLAMI-Entity-Matching-Dataset).
6. M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” TMLR 2024. [Paper](https://arxiv.org/abs/2304.07193).
