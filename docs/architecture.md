# Architecture

## Components

- `ml/datasets/abo.py` retrieves genuine ABO listing records and their referenced small catalog images.
- `ml/datasets/handbag_filter.py` applies the conservative product-type/title rule for handbag, purse and fashion-tote scope.
- `scripts/refine_handbag_dataset.py` joins current local rows to source listing metadata, validates image bytes, audits exact hashes and category/identity counts, writes the refined manifest, and creates identity-disjoint splits.
- `ml/datasets/pairs.py` samples identity-valid pairs and identity batches. Siamese negatives can be sampled from pretrained visual neighbors; triplet batches support hard or random negative selection.
- `ml/models/siamese.py` contains shared ResNet50 metric-learning models.
- `scripts/train.py`, `scripts/train_triplet.py` save resumable best/last checkpoints and epoch histories.
- `scripts/evaluate.py` selects thresholds on validation data and writes held-out pair and retrieval metrics, latency, model size and plots. `scripts/extract_dinov2.py` measures the frozen DINOv2 retrieval baseline.
- `scripts/build_index.py` writes catalog-order embeddings, FAISS index and metadata keyed to the source manifest and checkpoint hashes.
- `backend/` contains FastAPI, SQLAlchemy/SQLite records, strict index validation, upload validation, calibrated comparison and complete-link duplicate review.
- `frontend/` provides the React/TypeScript dashboard, search, compare, manual duplicate review, model performance and data audit pages.

## Data flow

ABO product records (`domain_name`, `item_id`) define stable marketplace-specific product identity. Main/alternate image IDs are joined to the official image metadata. The refined manifest retains source IDs, source titles/categories, paths, hashes, dimensions and license. Exact-hash duplicates across different identities are excluded as ambiguous; identity duplicates retain one copy. Splitting occurs by identity before pair/triplet construction.

## Index and database consistency

The embedding CSV, NumPy matrix and FAISS index are written in one fixed row order. Metadata stores source-manifest SHA-256, checkpoint SHA-256, catalog CSV hash, vector file hash, count, dimension and ordered identities/hashes. At startup, the service validates the current manifest, checkpoint, artifact hashes, row counts, dimensions and identity/hash order. A stale index is rejected, with a rebuild instruction, rather than being silently paired with another catalog. Database import writes the manifest hash used for synchronization; dashboard and search report database/index consistency.

## Inference and review

The API decodes only supported image formats, enforces upload size and minimum dimensions, and embeds through the loaded checkpoint. Search requires a valid compatible index. Compare returns a cosine similarity score; it includes a same/different decision only when calibration metadata matches the checkpoint hash. Batch review exposes every pair score. Complete-link clustering merges candidate groups only when all cross-group pair scores meet the selected validation threshold; uncalibrated runs return pairwise scores and no groups.

SQLite stores catalog records. Dataset images, reports, model weights and retrieval indexes remain local and are ignored by Git. Deployment requires persistent storage, a trusted CORS origin and suitable upload/rate limits.
