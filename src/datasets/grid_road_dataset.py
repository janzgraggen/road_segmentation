import logging

import torch

from src.datasets.image_road_dataset import ImageRoadDataset

logger = logging.getLogger(__name__)


class GridRoadDataset(ImageRoadDataset):
    """
    Treats the image as a grid of cells and returns each cell as a separate
    instance. Iterates over all images in sorted order and then over all cells
    in the image. It starts from the top-left corner, left to right and then top
    to bottom.

    Args:
        cell_size (int): size of the cell.
        image_size (int): size of the image.
        data_path (str): path to the images.
        target_path (str, optional): path to the masks. Defaults to None.
    """

    def __init__(
        self,
        cell_size: int,
        image_size: int,
        data_path: str,
        target_path: str | None = None,
    ):
        # Round the image size to the nearest multiple of the cell size
        image_size = (image_size // cell_size) * cell_size
        print(f"Rounded image size to {image_size}")

        assert image_size % cell_size == 0

        super().__init__(data_path, target_path)

        self.images = self._split_images(self.images, cell_size, image_size)

        if self.masks is not None:
            self.masks = self._split_images(self.masks, cell_size, image_size)

            assert len(self.images) == len(self.masks)

    def _split_images(
        self,
        images: torch.Tensor,
        cell_size: int,
        image_size: int,
    ) -> torch.Tensor:
        """
        Splits the images into a grid of cells.
        """
        cells = []

        for image in images:
            for row in range(0, image_size, cell_size):
                for col in range(0, image_size, cell_size):
                    cell = image[:, row : row + cell_size, col : col + cell_size]
                    cells.append(cell.clone())

        return torch.stack(cells)
