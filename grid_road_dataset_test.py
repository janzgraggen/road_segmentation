import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.io

from src.datasets.grid_road_dataset import GridRoadDataset
from src.utils.train_utils import reconstruct

IMAGE_SIZE = 400
CELL_SIZE = 100

DATA_PATH = "road_data/train/images/"
TARGET_PATH = "road_data/train/groundtruth/"

GRID_LINE_COLOR = "red"

INDEX = 1
CELL_INDEX = 5


if __name__ == "__main__":
    dataset = GridRoadDataset(CELL_SIZE, IMAGE_SIZE, DATA_PATH, TARGET_PATH)

    grid_size = IMAGE_SIZE // CELL_SIZE
    cells = grid_size**2

    sub_images = []
    sum_masks = []

    for i in range(cells):
        data = dataset[INDEX * cells + i]

        sub_images.append(data["img"])
        sum_masks.append(data["mask"])

    image = reconstruct(sub_images, grid_size)
    cell_mask = reconstruct(sum_masks, grid_size)

    # Load the original image and mask and assert that they are the same
    image_name = f"satImage_{INDEX + 1:03}.png"
    image_original = torchvision.io.read_image(DATA_PATH + image_name)
    mode = torchvision.io.ImageReadMode.GRAY
    mask_original = torchvision.io.read_image(TARGET_PATH + image_name, mode)

    assert torch.allclose(image, image_original)
    assert torch.allclose(cell_mask, mask_original)

    image = image.permute(1, 2, 0).numpy()
    cell_mask = cell_mask.permute(1, 2, 0).numpy()

    xticks = np.arange(CELL_SIZE, IMAGE_SIZE, CELL_SIZE)
    yticks = np.arange(CELL_SIZE, IMAGE_SIZE, CELL_SIZE)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Reconstructed Image")

    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True, color=GRID_LINE_COLOR)

    plt.subplot(2, 2, 2)
    plt.title("Reconstructed Ground Truth")
    plt.imshow(cell_mask, cmap="gray")

    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(True, color=GRID_LINE_COLOR)

    # Visualize a single cell
    cell = dataset[INDEX * cells + CELL_INDEX]
    cell_image = cell["img"].permute(1, 2, 0).numpy()
    cell_mask = cell["mask"].permute(1, 2, 0).numpy()

    plt.subplot(2, 2, 3)
    plt.imshow(cell_image)
    plt.title(f"Cell Image ({CELL_INDEX})")

    plt.subplot(2, 2, 4)
    plt.title(f"Cell Ground Truth ({CELL_INDEX})")
    plt.imshow(cell_mask, cmap="gray")

    plt.tight_layout()
    plt.show()
