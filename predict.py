# import the necessary packages
from xml.parsers.expat import model

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import metrics as mt


def prepare_plot(origImage, origMask, predMask, number, save=False):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	#figure.show()
	figure.savefig(f"output/prediction_{number}.png")
	

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH,
									config.INPUT_IMAGE_HEIGHT))
		gtMask = (gtMask > 0).astype(np.uint8)
        
        # make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		# ✅ binarisation propre
		predMask = (predMask > config.THRESHOLD).astype(np.uint8)
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask, number=filename.split(".")[0], save=True)

		#Metrics calculation
		dice = mt.dice_coefficient(gtMask, predMask)
		sdice = mt.surface_dice(gtMask, predMask, tolerance_mm=5.0)
		assd = mt.average_symmetric_surface_distance(gtMask, predMask)
		print(f"Metrics for {filename} - Dice: {dice:.4f}, Surface Dice@5mm: {sdice:.4f}, ASSD: {assd:.4f}")

	return dice, sdice, assd
		

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=len(os.listdir(config.TEST_FINAL))) # len(os.listdir(config.TEST_FINAL))
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)
# iterate over the randomly selected test image paths
dice_list = []
sdice_list = []
assd_list = []
for path in imagePaths:
	# make predictions and visualize the results
	dice, sdice, assd = make_predictions(unet, path)
	dice_list.append(dice)
	sdice_list.append(sdice)
	assd_list.append(assd)

# print the average metrics across all test images
print(f"Average Dice: {np.mean(dice_list):.4f}, Average Surface Dice@5mm: {np.mean(sdice_list):.4f}, Average ASSD: {np.mean(assd_list):.4f}")