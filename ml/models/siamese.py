from __future__ import annotations
import torch
from torch import nn
from torchvision import models

class EmbeddingBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        features = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        self.projection = nn.Sequential(nn.Linear(features, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x):
        return nn.functional.normalize(self.projection(self.encoder(x)), p=2, dim=1)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = False):
        super().__init__()
        self.backbone = EmbeddingBackbone(embedding_dim, pretrained)
        self.embedding_dim = embedding_dim

    def embed(self, image): return self.backbone(image)
    def forward(self, image_a, image_b): return self.embed(image_a), self.embed(image_b)

class ContrastiveLoss(nn.Module):
    """Labels are 1 for the same product and 0 for different products."""
    def __init__(self, margin: float = 1.0):
        super().__init__(); self.margin = margin
    def forward(self, a, b, same):
        distance = torch.linalg.vector_norm(a - b, dim=1)
        same = same.float()
        return (same * distance.square() + (1 - same) * torch.relu(self.margin - distance).square()).mean()

class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = False):
        super().__init__(); self.backbone = EmbeddingBackbone(embedding_dim, pretrained)
    def embed(self, image): return self.backbone(image)
    def forward(self, anchor, positive, negative):
        return self.embed(anchor), self.embed(positive), self.embed(negative)
