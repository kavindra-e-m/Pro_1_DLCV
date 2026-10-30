# Dataset documentation

## Amazon Berkeley Objects

Primary source: [official ABO page](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html). The official page identifies CC BY 4.0, 147,702 listings, 398,212 unique catalog images, a listing archive and a 3 GB small-image archive. This implementation downloads the listing archive and the 6 MB official image metadata table, filters listing metadata locally, then fetches the referenced small-image paths. Image bytes are never synthesized.

The bag filter examines product type, item name, keywords and category paths for handbag, shoulder bag, tote, purse, backpack, satchel, clutch, crossbody, messenger bag, duffel, luggage and bag terms. This is a broad category query, not a perfect handbag classifier. Review actual category distribution before a research claim.

The normalized manifest includes source dataset, `domain_name:item_id`, stable product identity, category, brand, local path, original image ID/name, original source URL, SHA-256, dimensions and license. Original catalog IDs are preserved. Raw downloads and manifests are ignored by Git.

Attribution is required to Amazon.com and the ABO dataset authors: Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Arnab Dhua, Xi (Brian) Zhang, Tomas Yago-Vicente, Jasmine Collins, Shubham Goel and Jitendra Malik. Cite “ABO: Dataset and Benchmarks for Real-World 3D Object Understanding,” CVPR 2022.

## GLAMI Entity Matching

The current [Hugging Face dataset card](https://huggingface.co/datasets/zidcenek/GLAMI-Entity-Matching-Dataset) documents `images` keyed by `itemId`, item metadata (`itemId`, price, color tags, department IDs, brand edition tag, title, description, geo and a train `label`), train/test item tables, train/validation splits, and group CSVs (`item1..itemN`, `label`) with hard/category variants. It reports Apache 2.0 and a 17.8 GB repository. Its label production section is still marked TODO. No GLAMI labels are silently treated as definitive product identities by this implementation.

## DeepBag and Stanford Online Products

The DeepBag paper describes 5,545 images and 220 handbag models but the project could not verify a public authorized download of its original dataset. The paper is cited as prior work only. Stanford Online Products is a general product retrieval dataset, not handbag-only, and is not included in current local counts.

## Reports and split protocol

- `data/reports/dataset_acquisition.json`: matching records, image candidate count, decoded image count, errors, source and license.
- `data/reports/data_quality_report.json`: input, valid, corrupted and exact duplicate counts.
- `data/reports/dataset_statistics.json`: valid image/identity counts, per-identity image counts, possible positive/negative pair counts, category and source distributions.
- `data/splits/{train,validation,test}.csv`: reproducible identity-level split. No product identity may appear in more than one split.

Counts are computed after actual download and decoding. A capped run is a sample of the sorted matching listing records, not a representative benchmark by default. Record command arguments and seed in any report.
