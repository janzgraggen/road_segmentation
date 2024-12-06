import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        smooth = 1

        probs = torch.sigmoid(logits)

        intersection = torch.sum(probs * labels, dim=1)
        union = torch.sum(probs, dim=1) + torch.sum(labels, dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return {"loss": (1 - dice).mean()}
