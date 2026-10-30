# Model methodology

## Preprocessing

Images are decoded with Pillow, normalized to RGB, checked for at least 32×32 pixels, and resized to 224×224 for model input. Training uses resized random crops, horizontal flips, modest color jitter and ImageNet channel normalization. Augmentations transform source photos only during training and are not counted as additional images. No object detector is claimed.

## ResNet50 baseline

The baseline exposes an ImageNet-pretrained ResNet50 global-average-pooled 2048-dimensional feature vector, L2-normalized for cosine retrieval. It is a pretrained feature extractor, not a BagVision-fine-tuned classifier.

## Siamese model

Both image inputs pass through one shared ResNet50 backbone and a projection head to configurable normalized embeddings. For same-product label `y=1`, contrastive loss penalizes squared embedding distance. For different-product label `y=0`, it penalizes distances below margin `m`: `y·d² + (1-y)·max(0,m-d)²`. Pair generation takes positive examples only from the same source product identity. Negatives draw from distinct identities. The validation/test datasets remain grouped by product identity.

Training uses AdamW, ReduceLROnPlateau, fixed seeds, early stopping and separate best/last checkpoints. `--resume` restores optimizer state from the last checkpoint.

## Triplet alternative

Triplets contain an anchor and positive from the same product identity and a negative from another product identity. The current experimental implementation uses randomly selected negative identities with PyTorch triplet margin loss. It does not claim semi-hard mining. That can be added and compared after baseline acquisition and run-time measurements.

## Evaluation

The threshold is selected to maximize validation F1, then frozen for test evaluation. Reported pair metrics include accuracy, precision, recall, F1, ROC AUC, average precision and confusion matrix. The dataset is identity-disjoint across the three partitions. The evaluation unit is a pair, and pair sampling is not an estimate of production prevalence.

## Modern encoder

The optional DINOv2-base extractor uses the pretrained `facebook/dinov2-base` checkpoint with Transformers and reads the test split only. It saves recall@1/5/10 and mean average precision for within-test product retrieval. The official model card and repository identify the weights as Apache 2.0. This feature extractor is frozen; it is not fine-tuned.
