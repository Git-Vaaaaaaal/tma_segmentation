import torch
import cv2
import numpy as np
from huggingface_hub import hf_hub_download, login
from PIL import Image
import config
from model_resnet import UNetResNet


#Main variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_HEIGHT = 1024


def load_model_from_hf(repo_id, filename, device):
    model = UNetResNet()

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

def predict_mask(imagePath, output_path):

    model = load_model_from_hf(
    repo_id="Vaaaal/TMAs",
    filename="unet_tgs_salt_resnet.pth",
    device=config.DEVICE
    )

    with torch.no_grad():

        # --- Load image (robuste TIFF) ---
        try:
            image = np.array(Image.open(imagePath).convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"Erreur chargement image: {imagePath}") from e

        origH, origW = image.shape[:2]

        # --- Preprocess ---
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

        # --- Post-process (important) ---
        predMask = torch.sigmoid(predMask)  # si binaire
        predMask = predMask.squeeze().cpu().numpy()

        # resize back original size
        predMask = cv2.resize(predMask, (origW, origH))

        # threshold (optionnel)
        predMask = (predMask > 0.5).astype("uint8") * 255

        Image.fromarray(predMask).save(output_path, compression=None)

    return predMask
