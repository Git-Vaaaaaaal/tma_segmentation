# TMAx 
This project proposes an image segmentation model for histopathological images, specifically designed for Tissue MicroArrays (TMA) used in pathological anatomy.

The goal is to automatically segment structures of interest in digital slides, particularly in the context of lymphomas.

# Documentation
def predict_mask(model, imagePath, outputPath, repo_id="Vaaaal/TMAs", filename="tmas_segmenter.pth"):
- model (torch.nn.Module)
- Segmentation model (UNet architecture or equivalent) : Weights will be automatically loaded from Hugging Face.
- imagePath (str) : Path to the input image to be segmented.
- outputPath (str) : Path where the generated segmentation mask will be saved.
- repo_id (str, optional) : ID of the Hugging Face repository containing the model weights.
- filename (str, optional) : Name of the weight file in the repository.
