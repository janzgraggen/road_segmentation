import torch
from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCELoss(
            weight=torch.Tensor(
                [(46000 + 16000) / (2 * 46000), (46000 + 16000) / (2 * 16000)]
            )
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {"loss": self.loss(logits, labels)}
