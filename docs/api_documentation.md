# API documentation

Base URL: `http://localhost:8000/api`. Interactive OpenAPI is at `/docs` while the server runs.

| Method | Endpoint | Inputs / result |
|---|---|---|
| GET | `/health` | Service, model, device and index availability. |
| GET | `/dashboard` | SQLite catalog counts, model availability and saved metrics. |
| POST | `/search` | Multipart image field `file`, optional `limit`; ranked results. |
| POST | `/compare` | Multipart fields `image_a`, `image_b`; cosine score and threshold decision. |
| POST | `/duplicate-detection` | Multipart repeated `files`, 2–100 images; candidate groups. |
| GET | `/products` | Optional `q`, `limit`, `offset`. |
| GET | `/products/{id}` | Product provenance and image metadata. |
| GET | `/model/metrics` | Saved evaluation artifact or empty object. |
| GET | `/dataset/statistics` | Generated dataset report or not-prepared status. |

Uploads accept JPEG, PNG, WebP and BMP up to 15 MB by default, and must decode to at least 32×32 pixels. Inference endpoints return 503 if the trained checkpoint is absent. Catalog retrieval returns no results until the index is built. Images are served from `/media/abo/` after local preparation.
