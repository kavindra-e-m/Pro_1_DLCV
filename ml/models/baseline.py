from torch import nn
from torchvision import models

class ResNet50Baseline(nn.Module):
    """ImageNet-pretrained ResNet50 global-average pooled feature baseline."""
    def __init__(self, pretrained=True):
        super().__init__(); weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.network=models.resnet50(weights=weights); self.network.fc=nn.Identity()
    def forward(self, image):
        import torch.nn.functional as F
        return F.normalize(self.network(image),dim=1)
