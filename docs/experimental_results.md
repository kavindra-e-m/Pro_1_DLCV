# Experimental results

## Initial capped ABO experiment

- Acquisition: `python -m scripts.prepare_data --max-products 250 --max-images-per-product 4`; exact ABO product IDs and image URLs were retained. After validation and cross-identity duplicate exclusion: 855 valid images, 235 product identities, 0 corrupted images, 31 ambiguous exact-image hash groups excluded. Category counts: HANDBAG 211, BACKPACK 184, BAG 53, LUGGAGE 52, SUITCASE 263, WALLET 55, TOTE_BAG 19 and CAMERA_BAGS_AND_CASES 18 images. These categories include adjacent bag/luggage types.
- Split: seed 42; train 188 identities / 681 images, validation 23 / 86, test 24 / 88. Product identities are disjoint between partitions.
- Siamese run: official ImageNet-pretrained ResNet50; 256-dimensional normalized embedding; contrastive loss; one epoch; batch size 8; AdamW learning rate 0.0001; CPU; seed 42. Training loss 0.51474; validation loss 0.30433. This short run validates the pipeline and is not a converged model.
- Threshold: 0.33107, selected by maximum F1 on 149 validation pairs.
- Held-out test: 152 pairs (64 positive, 88 negative). Accuracy 0.6974; precision 0.6154; recall 0.7500; F1 0.6761; ROC AUC 0.7711; average precision 0.6854. Confusion matrix: TN 58, FP 30, FN 16, TP 48.
- Triplet comparison: one epoch, batch size 16, ImageNet ResNet50 encoder frozen while the projection head trained; train triplet loss 0.12923, validation triplet loss 0.13473. Its separately selected validation threshold was 0.07308. On the same 152 test pairs: accuracy 0.7763; precision 0.6923; recall 0.8438; F1 0.7606; ROC AUC 0.8686; average precision 0.8112. Confusion matrix: TN 64, FP 24, FN 10, TP 54.
- Frozen DINOv2-base retrieval: official `facebook/dinov2-base` Apache-2.0 weights, no fine-tuning. On 88 test images, 86 queries had another same-identity test image; recall@1 0.3256, recall@5 0.6744, recall@10 0.8488, mAP 0.2878. This is a retrieval metric and is not directly comparable to the pair-classification numbers above.
- Limitations: small capped sample; catalog imagery only; broad bag/luggage categories; one epoch; pair construction and class prevalence are not representative of deployment. Do not interpret these values as expected performance on user photographs.

The local JSON, CSV and curve artifacts are ignored by Git. They were generated from the run above and are not fabricated values.

For future runs, include:

- Exact ABO acquisition cap and image cap, generated valid image and identity counts, source/license, and data-quality counts.
- Split seed and identity counts per split.
- Pretrained initialization and checkpoint source.
- Batch size, learning rate, epochs, early stopping, device and wall-clock training time.
- Validation threshold, positive/negative pair counts, test accuracy/precision/recall/F1/ROC AUC/average precision, and confusion matrix.
- Exact command, code revision and any limitations of catalog-style imagery.

The app reads `artifacts/metrics/evaluation.json` and training history. It does not display placeholder values.
