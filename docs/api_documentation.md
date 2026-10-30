# API documentation

Base URL: `http://localhost:8000/api`. Interactive OpenAPI is at `/docs` while the server runs.

| Method | Endpoint | Inputs / result |
|---|---|---|
| GET | `/health` | Service, model, device, calibration state and index validation/error. |
| GET | `/dashboard` | SQLite catalog counts, model availability, saved metrics and DB/manifest/index consistency. |
| POST | `/search` | Multipart image field `file`, optional `limit`; ranked results if the index is current. |
| POST | `/compare` | Multipart fields `image_a`, `image_b`; cosine score plus a decision only when checkpoint-matched calibration exists. |
| POST | `/duplicate-detection` | Multipart repeated `files`, 2–100 images; all pair scores and conservative complete-link groups when calibrated. |
| GET | `/products` | Optional `q`, `limit`, `offset`. |
| GET | `/products/{id}` | Product provenance and image metadata. |
| GET | `/model/metrics` | Saved evaluation artifact or empty object. |
| GET | `/dataset/statistics` | Generated dataset report or not-prepared status. |

Uploads accept JPEG, PNG, WebP and BMP up to 15 MB by default, and must decode to at least 32×32 pixels. Search returns 503 if the checkpoint or a hash-validated index is unavailable. Missing calibration yields raw similarity only; it never uses a hardcoded threshold. Duplicate grouping requires every proposed member pair to clear the validation threshold, avoiding connected-component transitivity. Images are served from `/media/abo/` after local preparation.
