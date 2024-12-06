import torch

from src.metrics.base_metric import BaseMetric


class F1Metric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Accuracy Metric
        """
        super().__init__(*args, **kwargs)

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
        predictions = (proba > self.threshold).float()

        tp = (predictions * labels).sum()
        # tn = ((1 - predictions) * (1 - labels)).sum()
        fp = (predictions * (1 - labels)).sum()
        fn = ((1 - predictions) * labels).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1
