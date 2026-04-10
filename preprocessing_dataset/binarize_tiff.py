from PIL import Image
import numpy as np
import os

def binarize_tiff(input_path, output_path, threshold=128):
    img = Image.open(input_path).convert("L")  # niveau de gris
    arr = np.array(img)
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)  # binaire
    Image.fromarray(arr).save(output_path, compression=None)

path = "dataset_raw"
mask = "mask"

marker = ["BCL2", "BCL6", "CD10", "MUM1", "MYC", "HE"]
for m in marker :
    list_img = os.listdir(os.path.join(path, mask, m))
    for img in list_img :
        input_path = os.path.join(path, mask, m, img)
        output_path = os.path.join(path, mask, m, img)
        binarize_tiff(input_path, output_path)
        print(f"{input_path} binarized")