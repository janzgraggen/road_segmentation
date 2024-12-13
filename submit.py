import csv
import os

import hydra
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from src.datasets.data_utils import get_dataloaders
from src.utils.io_utils import ROOT_PATH
from src.utils.train_utils import reconstruct


@hydra.main(version_base=None, config_path="src/configs", config_name="submit")
def main(config):
    name = "model_best.pth"
    path = os.path.join(ROOT_PATH, config.state_dir, name)

    # Load the best model state
    print(f"Loading model state from {path}")
    state = torch.load(path, weights_only=False)

    state_config = state["config"]

    # Set the device
    if state_config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = state_config.trainer.device

    print(f"Using device: {device}")

    # Instantiate the model
    model = instantiate(state_config.model)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Get the image and patch size
    image_size = config.datasets.test.image_size
    patch_size = state_config.model.patch_size

    # assert image_size % patch_size == 0

    grid_size = image_size // patch_size

    # Get the dataloaders and batch transforms
    config.datasets.test.cell_size = patch_size
    print("Getting dataloaders...")
    dataloaders, batch_transforms = get_dataloaders(config, device)
    print("Dataloaders loaded successfully")

    dataloader = dataloaders["test"]
    transform = batch_transforms["test"]

    print("Starting inference...")
    # Run inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            transformed = transform(batch)
            output = model(transformed["img"])
            logits = output["logits"]
            proba = torch.nn.functional.sigmoid(logits)
            prediction = proba > 0.5  # TODO: Get this value from the config...
            predictions.append(prediction.int())

    # Concatenate the predictions
    predictions = torch.cat(predictions)

    predictions = predictions.reshape(-1, patch_size // 16, patch_size // 16)

    assert len(predictions) % (grid_size**2) == 0

    # Split the predictions into images to reconstruct
    images = torch.split(predictions, grid_size**2)

    assert len(images) == 50

    # Reconstruct the images
    reconstructed_images = [reconstruct(image, grid_size) for image in images]

    submission = []
    for i, image in enumerate(reconstructed_images):
        image = image.squeeze().cpu()

        for j, row in enumerate(image):
            for k, patch in enumerate(row):
                id = f"{i+1:03d}_{j * 16}_{k * 16}"
                submission.append([id, patch.item()])

        # for w in range(38):
        #     for h in range(38):
        #         id = f"{i+1:03d}_{w * 16}_{h * 16}"
        #         submission.append([id, image[h, w].item()])

    name = "submission.csv"
    submission_path = os.path.join(ROOT_PATH, config.state_dir, name)

    with open(submission_path, "w", newline="") as f:
        columns = ["id", "prediction"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for id, prediction in submission:
            writer.writerow({"id": id, "prediction": prediction})

    print(f"Submission file saved at {submission_path}")


if __name__ == "__main__":
    main()
