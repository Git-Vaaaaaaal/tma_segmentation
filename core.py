import torch
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from . import config

def load_model_from_hf(model, repo_id, filename, device):
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model"
    )

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model


def predict_and_save_mask(model, imagePath, outputPath, repo_id=None, filename=None):
    model.eval()

    # --- load weights HF (optionnel) ---
    if repo_id is not None and filename is not None:
        model = load_model_from_hf(model, repo_id, filename, config.DEVICE)

    with torch.no_grad():
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        origH, origW = image.shape[:2]

        image_resized = cv2.resize(
            image,
            (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        )

        image_resized = image_resized.astype("float32") / 255.0
        image_resized = np.transpose(image_resized, (2, 0, 1))
        image_resized = np.expand_dims(image_resized, 0)

        image_tensor = torch.from_numpy(image_resized).to(config.DEVICE)

        # --- Prediction ---
        predMask = model(image_tensor)

    return predMask
