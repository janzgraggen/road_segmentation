"""
This script moves a fraction of the training data to the validation data 
directory.
"""

import os

import numpy as np

TRAIN_DATA_PATH = "road_data/train/"

IMAGES = "images/"
GROUND_TRUTH = "groundtruth/"

VALIDATION_DATA_PATH = "road_data/validation/"

SPLIT = 0.1

if __name__ == "__main__":
    np.random.seed(42)

    # Create the validation data directory if it does not exist
    os.makedirs(VALIDATION_DATA_PATH, exist_ok=True)
    os.makedirs(VALIDATION_DATA_PATH + IMAGES, exist_ok=True)
    os.makedirs(VALIDATION_DATA_PATH + GROUND_TRUTH, exist_ok=True)

    # List the training data
    images = os.listdir(TRAIN_DATA_PATH + IMAGES)
    masks = os.listdir(TRAIN_DATA_PATH + GROUND_TRUTH)

    assert len(images) == len(masks)

    split = int(len(images) * SPLIT)

    indices = np.random.permutation(len(images))

    for i in indices[:split]:
        image = images[i]
        mask = masks[i]

        print(f"Moving {image} to validation data")

        os.rename(
            TRAIN_DATA_PATH + IMAGES + image, VALIDATION_DATA_PATH + IMAGES + image
        )
        os.rename(
            TRAIN_DATA_PATH + GROUND_TRUTH + mask,
            VALIDATION_DATA_PATH + GROUND_TRUTH + mask,
        )
