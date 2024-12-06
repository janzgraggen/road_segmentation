import torch


def create_target(
    mask: torch.Tensor, patch_size: int, road_threshold: float
) -> torch.Tensor:
    """
    Create target from mask.

    Args:
        mask (torch.Tensor): binary mask of shape (H, W).
    Returns:
        target (torch.Tensor): target of shape (H // patch_size, W // patch_size).
    """

    # Create a patch size kernel of ones to convolve with the mask
    # The kernel has one batch and one channel dimension
    kernel = torch.ones(1, 1, patch_size, patch_size, device=mask.device)

    # Convolve the mask with the kernel to get the number of roads in the patch
    with torch.no_grad():
        roads = torch.nn.functional.conv2d(mask, kernel, stride=patch_size)

    # Threshold the number of roads to get the target
    threshold = road_threshold * patch_size**2
    target = roads > threshold

    return target.flatten(1).float()


def reconstruct(images: list[torch.Tensor], grid_size: int) -> torch.Tensor:
    """
    Reconstructs the images into from a grid of cells.

    Args:
        images (list[torch.Tensor]): list of images.
        grid_size (int): size of the grid.

    Returns:
        reconstructed (torch.Tensor): reconstructed image.
    """
    assert len(images) == grid_size**2

    rows = []
    for row in range(grid_size):
        columns = []
        for col in range(grid_size):
            index = row * grid_size + col
            columns.append(images[index])

        row = torch.cat(columns, dim=-1)
        rows.append(row)

    reconstructed = torch.cat(rows, dim=-2)

    return reconstructed
