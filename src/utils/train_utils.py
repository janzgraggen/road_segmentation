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

    # Remove batch and channel dimension
    roads = roads.squeeze()

    # Threshold the number of roads to get the target
    threshold = road_threshold * patch_size**2
    target = roads > threshold

    return target.flatten(1).float()
