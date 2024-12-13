import argparse
import csv
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from src.datasets.image_road_dataset import ImageRoadDataset
from src.utils.io_utils import ROOT_PATH
from src.utils.train_utils import create_target


def main(image_number, validation=False):
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
        if img_num == image_number + 1:
            w = width // patch_size
            h = height // patch_size
            pred_image[h][w] = label

    if validation:
        print("Using validation dataset")
        dataset = ImageRoadDataset(
            os.path.join(ROOT_PATH, "road_data", "validation", "images"),
            os.path.join(ROOT_PATH, "road_data", "validation", "groundtruth"),
        )
    else:
        print("Using test dataset")
        dataset = ImageRoadDataset(
            os.path.join(ROOT_PATH, "road_data", "aicrowd"),
        )

    image = dataset[image_number]["img"]

    # Transforming image to numpy array
    image = image.permute(1, 2, 0).numpy()

    # Visualizing the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Visualizing the mask
    if validation:
        mask = dataset[image_number]["mask"]

        # Add a batch dimension and scale to [0, 1]
        mask = mask.unsqueeze(0) / 255

        # Create the target from the mask, this will be a flattened tensor
        target = create_target(mask, 16, 0.25).numpy()

        # Reshape the target to the image size, we assume the image is square
        width = int(target.size**0.5)
        target = target.reshape(width, width)

        # Visualizing the mask in green
        for i in range(len(target)):
            for j in range(len(target[i])):
                if target[i][j] == 1:
                    rect = patches.Rectangle(
                        (j * patch_size, i * patch_size),
                        patch_size,
                        patch_size,
                        facecolor="g",
                        alpha=0.4,
                    )
                    ax.add_patch(rect)

    # Visualizing the predictions in red
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
    ap.add_argument("-v", "--validation", action="store_true")
    args = ap.parse_args()

    main(args.image, args.validation)
