import segmentation_models_pytorch as smp
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {"loss": self.loss_fn(logits, labels)}
