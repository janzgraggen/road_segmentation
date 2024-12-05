import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Accuracy Metric
        """
        super().__init__(*args, **kwargs)
        self.threshold = 0.5

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            accuracy (float): calculated metric.
        """

        proba = torch.nn.functional.sigmoid(logits)
        predictions = proba > self.threshold
        return (predictions == labels).mean(dtype=torch.float32)
