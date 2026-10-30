# Architecture

## Components

- `ml/datasets/abo.py` reads authentic ABO listings, filters bags using the release metadata, joins listed image IDs to the official image metadata, downloads only selected small catalog images, and writes provenance to a CSV manifest.
- `scripts/dataset_cleaning.py` decodes and dimensions-checks images, hashes bytes, records corrupted/duplicate counts and updates the manifest.
- `scripts/dataset_split.py` partitions identities first, then writes every image for each identity to exactly one of train, validation or test.
- `ml/models/siamese.py` shares a ResNet50 encoder and projects to normalized embeddings. Contrastive and triplet losses are available.
- `scripts/evaluate.py` calibrates a threshold on validation pairs and calculates held-out test pair metrics and plots.
- `scripts/build_index.py` extracts embeddings for valid catalog images and writes a FAISS inner-product index when FAISS is installed; exact NumPy cosine search remains available without FAISS.
- `backend/` contains FastAPI endpoints, SQLAlchemy/SQLite models and a checkpoint-backed inference service.
- `frontend/` is a React, TypeScript and Vite client with dashboard, retrieval, pair comparison, batch review, catalog, model and dataset views.

## Data flow

ABO product record (`domain_name`, `item_id`) becomes a stable identity `ABO:<domain>:<item_id>`. Its main and alternate image IDs are joined through ABO image metadata to small-image file paths. Source path, license, URL, width, height, hash and identity are preserved in the manifest. Cleaning removes invalid records from the training manifest and records counts. Identity-grouped splitting occurs before any pair generation.

## Inference

The API loads the saved Siamese checkpoint if it exists. Images are decoded as RGB, resized to 224×224, normalized with ImageNet statistics and embedded. Search uses inner products between normalized vectors (cosine similarity). The compare and batch endpoints use the threshold saved from validation when available. Without a checkpoint, inference endpoints return HTTP 503 instead of producing a fabricated score.

## Storage and deployment notes

SQLite stores imported product/image metadata. Images, reports, indexes and checkpoints remain local and are ignored by Git. CORS allows the local Vite dev origin. For deployment, configure an explicit trusted origin, persistent storage and request limits appropriate to the host; do not expose local paths or upload directories publicly.
