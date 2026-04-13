<img width="2242" height="1267" alt="image" src="https://github.com/user-attachments/assets/3e0d6e0b-aafa-400b-a129-63b837c2ee01" />

# TMAs 
Ce projet propose un modèle de segmentation d’images histopathologiques, spécialisé dans les Tissue MicroArrays (TMA) utilisés en anatomopathologie.

L’objectif est de segmenter automatiquement des structures d’intérêt dans des lames numériques, en particulier dans le contexte des lymphomes.

# Documentation
def predict_mask(model, imagePath, outputPath, repo_id="Vaaaal/TMAs", filename="tmas_segmenter.pth"):
- model (torch.nn.Module)
- Segmentation model (UNet architecture or equivalent) : Weights will be automatically loaded from Hugging Face.
- imagePath (str) : Path to the input image to be segmented.
- outputPath (str) : Path where the generated segmentation mask will be saved.
- repo_id (str, optional) : ID of the Hugging Face repository containing the model weights.
- filename (str, optional) : Name of the weight file in the repository.
