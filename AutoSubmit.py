
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
    NBR_IMGS = 50
    IMGSIZE = 608  
    CROP_SIZE = 304 #16
    assert(IMGSIZE % CROP_SIZE == 0)
    assert(CROP_SIZE % 16 == 0)
    PREDICT_SIZE = CROP_SIZE/16 # 19 , #1 
    assert(PREDICT_SIZE*crop_factor == 38) #predictions per image
 
    crop_factor = IMGSIZE // CROP_SIZE # 2 #38 # = Predictions per image row
    predicts_per_image = crop_factor**2

    

    for i in range(NBR_IMGS):
        #default torch tensor of size 38x38
        pred_one_img_tensor = torch.zeros(38,38)

        for j in range(crop_factor):
            for k in range(crop_factor):
                #prediction index: (index of the predicted units (Matrix'es w/shape PRED_SIZE x PRED_SIZE)
                pred_idx = i * predicts_per_image +j * crop_factor +k
                prediction = torch.load(save_path / f"output_{pred_idx}.pth")
                
                pred_one_img_tensor[j,k] = prediction["pred_label"].item() ## VERIFY IF THIS IS CORRECT
        #Pred_1 img TO TENSOR is 38x38
        for p in range(38):
            for q in range(38):
                pred_patch = pred_one_img_tensor[j,k]
                identifier = f"{'%.3d' % (i + 1)}_{p * 16}_{q * 16}"
                submission.append([identifier, pred_patch ])

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
