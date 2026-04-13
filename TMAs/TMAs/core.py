import torch
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import config
from model_resnet import UNetResNet


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


def load_model_local(path, device):
    model = UNetResNet()

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return model


def predict_mask(imagePath, model):

    with torch.no_grad():
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(
            image,
            (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        )

        image_resized = image_resized.astype("float32") / 255.0
        image_resized = np.transpose(image_resized, (2, 0, 1))
        image_resized = np.expand_dims(image_resized, 0)

        image_tensor = torch.from_numpy(image_resized).to(config.DEVICE)

        predMask = model(image_tensor)

    return predMask


model = load_model_from_hf(
    repo_id="Vaaaal/TMAs",
    filename="tmas_segmenter.pth",
    device=config.DEVICE
)

mask = predict_mask("09abe43a32f34fbb98d6ab10ff13ca16.tiff", model)