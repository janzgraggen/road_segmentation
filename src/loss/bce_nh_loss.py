import torch
import torch.nn as nn


class BinaryCrossEntropyLossWN(nn.Module):
    """
    Combined Binary Cross-Entropy Loss with Neighborhood Consistency Loss.
    """

    def __init__(self, lambda_neighborhood=1.0):
        """
        Initialize the loss function.

        Args:
            lambda_neighborhood (float): Weight for the Neighborhood Loss term.
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
        self.lambda_neighborhood = lambda_neighborhood

    def neighborhood_loss(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Compute the neighborhood consistency loss.

        Args:
            predictions (Tensor): Model predictions (logits).
            labels (Tensor): Ground truth labels.
        
        Returns:
            neighborhood_loss (Tensor): The calculated neighborhood consistency loss.
        """
        # Assuming `predictions` and `labels` are 2D tensors (grids)
        rows, cols = predictions.shape
        neighborhood_penalty = 0

        for r in range(1, rows - 1):  # Avoid border
            for c in range(1, cols - 1):  # Avoid border
                # Get the 3x3 neighborhood of the current pixel (r, c)
                patch = predictions[r - 1:r + 2, c - 1:c + 2]
                patch_labels = labels[r - 1:r + 2, c - 1:c + 2]
                
                # Count how many neighbors are roads (1)
                road_neighbors = (patch == 1).sum().item()
                
                if patch_labels[1, 1] == 1:  # Current patch is road
                    if road_neighbors < 2:  # Fewer than 2 neighbors are roads
                        neighborhood_penalty += 1
                else:  # Current patch is non-road
                    if road_neighbors > 6:  # More than 7 neighbors are roads
                        neighborhood_penalty += 1

        return neighborhood_penalty

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions (before sigmoid or softmax).
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        # Apply sigmoid to logits to get probabilities 
        probs = torch.sigmoid(logits)   

        # Compute Binary Cross-Entropy Loss
        bce_loss = self.loss(logits, labels)

        # Compute Neighborhood Loss
        neighborhood_loss = self.neighborhood_loss(probs, labels)

        # Combine losses with the lambda term
        total_loss = bce_loss + self.lambda_neighborhood * neighborhood_loss

        return {"loss": total_loss, "bce_loss": bce_loss, "neighborhood_loss": neighborhood_loss}

