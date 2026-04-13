import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import config

# ===== Decoder Block =====
class Block(nn.Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		self.conv1 = nn.Conv2d(inChannels, outChannels, 3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(outChannels, outChannels, 3, padding=1)

	def forward(self, x):
		return self.conv2(self.relu(self.conv1(x)))

# ===== Decoder =====
class Decoder(nn.Module):
	def __init__(self, channels):
		super().__init__()

		self.upconvs = nn.ModuleList([
			nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2)
			for i in range(len(channels)-1)
		])

		self.blocks = nn.ModuleList([
			Block(channels[i], channels[i+1])
			for i in range(len(channels)-1)
		])

	def forward(self, x, encFeatures):
		for i in range(len(self.upconvs)):
			x = self.upconvs[i](x)
			encFeat = encFeatures[i]

			# ajustement taille si besoin
			if x.shape != encFeat.shape:
				x = F.interpolate(x, size=encFeat.shape[2:])

			x = torch.cat([x, encFeat], dim=1)
			x = self.blocks[i](x)

		return x

# ===== UNet ResNet =====
class UNetResNet(nn.Module):
	def __init__(self, nbClasses=1):
		super().__init__()

		# ===== Encoder ResNet pré-entraîné =====
		resnet = resnet34(weights="IMAGENET1K_V1")

		self.input_block = nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu
		)
		self.input_pool = resnet.maxpool

		self.encoder1 = resnet.layer1
		self.encoder2 = resnet.layer2
		self.encoder3 = resnet.layer3
		self.encoder4 = resnet.layer4

		# ===== Decoder =====
		self.decoder = Decoder([512, 256, 128, 64])

		self.head = nn.Conv2d(64, nbClasses, 1)

	def forward(self, x):
		# ===== Encoder =====
		x1 = self.input_block(x)      # 64
		x2 = self.input_pool(x1)      # downsample
		x3 = self.encoder1(x2)        # 64
		x4 = self.encoder2(x3)        # 128
		x5 = self.encoder3(x4)        # 256
		x6 = self.encoder4(x5)        # 512

		# ===== Decoder =====
		x = self.decoder(x6, [x5, x4, x3, x1])

		# ===== Output =====
		x = self.head(x)
		x = F.interpolate(x, size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))

		return x
