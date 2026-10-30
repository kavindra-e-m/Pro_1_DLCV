# Model methodology

## Target and data protocol

The target is ABO catalog product identity (`domain_name:item_id`), not physical-object identity. Positives use distinct source image IDs attached to the same product record. Negative labels always come from another source product identity; visual similarity never turns a different product into a positive. Training, validation and test partitions are grouped by product identity before any pairs are formed.

ABO contains catalog imagery, not uncontrolled customer photography. The refined local experiment is small, selected and category-filtered; results should be treated as preliminary evidence for this data domain. No claim of real-world photo robustness is made.

## Preprocessing

Images are decoded as RGB, checked for at least 32×32 pixels, resized/cropped to 224×224, augmented during training with horizontal flip and modest color jitter, and normalized using ImageNet channel statistics. Augmentations do not add labeled photographs. No detector or segmentation model is claimed.

## Siamese ResNet50

Both inputs pass through shared ImageNet-initialized ResNet50 weights and a projection head to L2-normalized embeddings. Contrastive loss uses source identity labels: positive pairs minimize squared distance; negative pairs are pushed outside a margin. Pair sampling covers same-identity source image pairs and distinct-identity negatives. For the visual-negative condition, pretrained ResNet50 pooled features find visually close *other product identities*; top-ranked different identities are used as hard negative candidates. The random condition samples any other identity and provides the comparison baseline.

Training exposes configurable epochs/batch size, AdamW, ReduceLROnPlateau, fixed random seeds, validation-loss early stopping, `best` and `last` checkpoints, and optimizer/scheduler-state resumption. Device selection uses CUDA when available and falls back to CPU. The longer CPU run fine-tunes ResNet50's final residual block and projection head; earlier blocks remain frozen for runtime. This is backbone fine-tuning, not a frozen-encoder experiment. The exact run configuration is saved with the checkpoint. On CPU, full ResNet fine-tuning can be slow; report hardware and wall-clock duration with every run.

## Triplet ResNet50

Triplet experiments form identity-aware batches containing multiple distinct images per product. Batch-hard mode selects the farthest in-batch positive for each anchor and the nearest in-batch negative from a different product identity, optimizing triplet margin loss. Random mode samples a negative from another identity in the same batch for comparison. Identities with fewer than two images cannot contribute triplets. Validation uses deterministic identity batches and the same margin objective. The checkpoint records its mining mode and config.

## Calibration and evaluation

For each checkpoint, the cosine threshold is selected by maximizing F1 over validation pair scores; ties prefer precision and then the higher threshold. The calibration artifact stores the threshold, validation precision/recall/F1, selection method, UTC timestamp and SHA-256 of the checkpoint. Test identities do not choose the threshold. A missing or checkpoint-mismatched calibration produces a similarity score only and no definitive same/different prediction.

Pairwise test metrics include accuracy, precision, recall, F1, ROC AUC, PR-AUC/average precision and a confusion matrix. Retrieval metrics rank images within the identity-separated test corpus and report recall@1/5/10 and mAP. Latency is measured per image on the recorded device, and checkpoint/model-file size is reported. Siamese and triplet runs use the same refined split and deterministic pair-generation protocol. Frozen DINOv2 is a retrieval-only baseline; it is not fine-tuned and is not interpreted as a calibrated same/different classifier.

Pair metrics depend on the sampled positive/negative composition and are not an estimate of deployed prevalence. The limited number of test identities makes all estimates uncertain. A larger independently sourced handbag dataset and user-photo test set are needed before application claims.
