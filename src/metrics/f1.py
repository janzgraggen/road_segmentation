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

        logits = torch.nn.functional.softmax(logits, dim=1)
        predictions = logits > self.threshold

        tp = (logits * labels).sum().to(torch.float32)
        tn = ((1 - logits) * (1 - labels)).sum().to(torch.float32)
        fp = (logits * (1 - labels)).sum().to(torch.float32)
        fn = ((1 - logits) * labels).sum().to(torch.float32)
        return tp / (tp + 0.5 * (fp + fn))
