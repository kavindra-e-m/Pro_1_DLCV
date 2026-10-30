# Dataset documentation

## Source and target scope

The primary source is [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html), licensed CC BY 4.0. The acquisition pipeline joins real ABO product listings and image IDs to the official image metadata, then downloads only the referenced catalog images. No images or product labels are synthesized.

The target experiment includes ABO records with `HANDBAG` product type, plus generic `BAG` and `TOTE_BAG` records only when the actual listing title contains positive fashion-bag evidence (handbag, purse, tote, shoulder bag, clutch, satchel, evening bag, sling bag or crossbody). Explicit non-target product types such as backpack, luggage, suitcase, wallet, camera bag/case, briefcase, duffel and travel bag are excluded. Generic grocery/produce totes and laptop/backpack listings are filtered out. The source `domain_name:item_id` defines product identity; domain is included because item IDs are marketplace scoped.

## Current local audit (seed 42)

The audit was generated from the locally acquired bounded ABO subset and matching listing records:

- 855 acquired rows were reviewed; 248 valid catalog images from 66 product identities were retained.
- Retained categories: `HANDBAG` 211 images / 56 identities; title-verified fashion products from generic `BAG` 37 / 10. No `TOTE_BAG` records passed the fashion-vs-grocery title rule in this acquired subset.
- Excluded rows: `BACKPACK` 184; `SUITCASE` 263; `LUGGAGE` 52; non-fashion `BAG` 16; `CAMERA_BAGS_AND_CASES` 18; `WALLET` 55; grocery-style `TOTE_BAG` 19.
- All 263 retained images decoded and met the 32×32 minimum. No exact duplicate hashes were present in this retained subset. Two identities have one catalog image; the audit lists them explicitly. These are catalog images/views and are not claimed to be separate real-world photo sessions.
- Identity split, seed 42: train 52, validation 7, test 7 product identities (196/24/28 images). Pair/triplet training requires multiple retained source images for a positive match.
- Train, validation and test identity intersections are empty; the audit records each intersection and a boolean assertion.

The dataset is still small and selected from an earlier capped listing sample. It is not a representative sample of handbag brands or the marketplace. ABO catalog photos also differ from real user photographs in background, lighting, viewpoint, occlusion and image quality. Report the current sample size and category scope with every result.

## Audit and preparation outputs

- `data/processed/metadata/catalog_unfiltered.csv`: preserved pre-refinement local manifest.
- `data/processed/metadata/catalog.csv`: current handbag-target manifest with source title, source product type, category paths, source IDs, image URL, hash, dimensions and license.
- `data/reports/handbag_dataset_audit.json`: retained/excluded categories, per-identity image counts, duplicate hashes, identities with fewer than two images and identity split overlap checks.
- `data/reports/data_quality_report.json`: decoding, dimension and duplicate-image cleaning report.
- `data/reports/dataset_statistics.json`: cleaned counts, category distribution and embedded handbag audit.
- `data/splits/{train,validation,test}.csv`: deterministic identity-grouped splits.

Run `python -m scripts.prepare_data --max-products 250 --max-images-per-product 4 --seed 42` to recreate a bounded local run. Acquisition and filtering output depends on the complete source listing archive available locally; record the acquisition cap and seed when presenting metrics.

## Attribution and adjacent sources

Attribution is required to Amazon.com and the ABO dataset authors: Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Arnab Dhua, Xi (Brian) Zhang, Tomas Yago-Vicente, Jasmine Collins, Shubham Goel and Jitendra Malik. Cite “ABO: Dataset and Benchmarks for Real-World 3D Object Understanding,” CVPR 2022.

The [GLAMI Entity Matching dataset card](https://huggingface.co/datasets/zidcenek/GLAMI-Entity-Matching-Dataset) reports Apache 2.0 and a large multilingual product dataset, but its label-generation section is incomplete. It is documented only and is not included in local training. The DeepBag paper describes a handbag benchmark, but the project has not verified an authorized public download of its original images; it is cited as prior work only.
