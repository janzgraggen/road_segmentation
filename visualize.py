import argparse
import csv
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.datasets.data_utils import get_dataloaders
from src.utils.io_utils import ROOT_PATH
from src.utils.train_utils import reconstruct


def main(image_number):
    # Reading submission file
    submission_path = os.path.join(ROOT_PATH, "saved/testing", "submission.csv")
    with open(submission_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]

    patch_size = 16
    image_size = 608

    pred_image = [
        [0 for _ in range(image_size // patch_size)]
        for _ in range(image_size // patch_size)
    ]

    for i in range(len(data)):
        id, label = data[i]
        img_num, width, height = map(int, id.split("_"))
        if img_num == image_number:
            w = width // patch_size
            h = height // patch_size
            pred_image[h][w] = label

    # Loading image
    image_path = os.path.join(
        ROOT_PATH, "road_data", "aicrowd", f"test_{image_number}.png"
    )
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i in range(len(pred_image)):
        for j in range(len(pred_image[i])):
            if pred_image[i][j] == "1":
                rect = patches.Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size,
                    patch_size,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=int)
    args = ap.parse_args()

    main(args.image)
