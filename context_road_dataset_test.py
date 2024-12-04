import matplotlib.pyplot as plt
import numpy as np

from src.datasets.context_road_dataset import ContextRoadDataset
from src.transforms.custom_transform import CustomTransform

IMAGE_SIZE = 304
PATCH_SIZE = 16
ROAD_THRESHOLD = 0.25
DATA_PATH = "road_data/train/images/"
TARGET_PATH = "road_data/train/groundtruth/"

INDEX = 0

PATCH_LINE_COLOR = "red"

if __name__ == "__main__":
    dataset = ContextRoadDataset(DATA_PATH, TARGET_PATH)

    data = dataset[INDEX]

    transform = CustomTransform(IMAGE_SIZE, PATCH_SIZE, ROAD_THRESHOLD)
    data = transform(data)

    image = data["img"]
    mask = data["mask"]
    target = data["labels"]

    image = image * 255
    image = image.permute(1, 2, 0)
    image = image.int()
    image = image.numpy()

    mask = mask * 255
    mask = mask.permute(1, 2, 0)
    mask = mask.int()
    mask = mask.numpy()

    target_size = IMAGE_SIZE // PATCH_SIZE
    target = target.reshape(target_size, target_size)
    target = target.int()
    target = target.numpy()

    x_ticks = np.arange(0, IMAGE_SIZE, PATCH_SIZE)
    y_ticks = np.arange(IMAGE_SIZE, 0, -PATCH_SIZE)

    x_labels = np.arange(0, target_size, 1)
    y_labels = np.arange(target_size, 0, -1)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.xticks(x_ticks - 0.5, x_labels, rotation=90)
    plt.yticks(y_ticks - 0.5, y_labels)

    plt.grid(True, color=PATCH_LINE_COLOR)

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask, cmap="gray")
    plt.xticks(x_ticks - 0.5, x_labels, rotation=90)
    plt.yticks(y_ticks - 0.5, y_labels)

    plt.grid(True, color=PATCH_LINE_COLOR)

    plt.subplot(1, 3, 3)
    plt.imshow(target, cmap="gray")
    plt.title("Labels")

    plt.xticks(x_labels - 0.5, x_labels, rotation=90)
    plt.yticks(y_labels - 0.5, y_labels)

    plt.grid(True, color=PATCH_LINE_COLOR)

    plt.tight_layout()
    plt.show()
