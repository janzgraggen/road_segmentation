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

        logits = torch.nn.functional.sigmoid(logits)
        predictions = (logits > self.threshold).type(torch.float32)

        true_positives = torch.sum(predictions * labels)
        false_positives = torch.sum(predictions * (1 - labels))
        false_negatives = torch.sum((1 - predictions) * labels)

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1
