import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from src.datasets.data_utils import get_dataloaders
from src.metrics.f1 import F1Metric
from src.utils.io_utils import ROOT_PATH


def main(experiment: str):
    name = "model_best.pth"
    path = os.path.join(ROOT_PATH, "saved", experiment, name)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load the best model state
    print(f"Loading model state from {path}")
    state = torch.load(path, weights_only=False, map_location=device)

    state_config = state["config"]

    # Instantiate the model
    model = instantiate(state_config.model)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Get the dataloaders and batch transforms
    print("Getting dataloaders...")
    dataloaders, batch_transforms = get_dataloaders(state_config, device)
    print("Dataloaders loaded successfully")

    dataloader = dataloaders["inference"]
    transform = batch_transforms["inference"]

    metric = F1Metric()

    step = 0.05
    thresholds = np.arange(0, 1 + step, step)
    f1_scores = []

    print("Starting inference...")
    # Run inference
    with torch.no_grad():
        for threshold in thresholds:
            logits = []
            labels = []

            for batch in tqdm(dataloader, desc=f"{threshold:.2f}"):
                transformed = transform(batch)
                output = model(transformed["img"])
                logits.append(output["logits"])
                labels.append(transformed["labels"])

            logits = torch.concat(logits)
            labels = torch.concat(labels)

            metric.threshold = threshold
            f1_score = metric(logits, labels)
            f1_scores.append(f1_score.item())

    f1_csv_path = os.path.join(ROOT_PATH, "saved", experiment, "f1.csv")
    f1_plot_path = os.path.join(ROOT_PATH, "figures", f"{experiment}_threshold.png")

    with open(f1_csv_path, "w", newline="") as f:
        columns = ["threshold", "f1"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(len(thresholds)):
            writer.writerow({"threshold": thresholds[i], "f1": f1_scores[i]})

    print(f"F1 scores saved at {f1_csv_path}")

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(f1_plot_path)

    print(f"F1 score plot saved at {f1_plot_path}")

    # Find the best threshold
    best_f1_score = 0
    best_threshold = 0

    for i in range(len(thresholds)):
        if f1_scores[i] > best_f1_score:
            best_f1_score = f1_scores[i]
            best_threshold = thresholds[i]

    print(f"Best threshold {best_threshold:.2f} (f1 {best_f1_score:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=str)
    args = ap.parse_args()

    main(args.experiment)
