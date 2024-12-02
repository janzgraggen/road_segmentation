import csv
import os

import hydra
import numpy as np
import torch

from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path / "test"
    submission = []

    for i in range(50):
        for j in range(38):
            for k in range(38):
                l = i * 38 * 38 + j * 38 + k
                prediction = torch.load(save_path / f"output_{l}.pth")
                identifier = f"{'%.3d' % (i + 1)}_{k * 16}_{j * 16}"
                submission.append([identifier, prediction["pred_label"].item()])
    file_path = ROOT_PATH / "aicrowd" / "submission.csv"
    os.makedirs(ROOT_PATH / "aicrowd", exist_ok=True)

    with open(file_path, "w", newline="") as f:
        columns = ["id", "prediction"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for c1, c2 in submission:
            writer.writerow({"id": c1, "prediction": c2})

    print(f"Submission file saved at {file_path}")


if __name__ == "__main__":
    main()
