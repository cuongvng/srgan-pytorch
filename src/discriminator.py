import torch.nn as nn
import torch.nn.functional as F
import math

class Discriminator(nn.Module):
	def __init__(self, H, W):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv_blks = nn.Sequential(
			ConvBlock(64, 64, 2),
			ConvBlock(64, 128, 1),
			ConvBlock(128, 128, 2),
			ConvBlock(128, 256, 1),
			ConvBlock(256, 256, 2),
			ConvBlock(256, 512, 1),
			ConvBlock(512, 512, 2)
		)
		H_out, W_out = self._get_HW_out(H, W)
		self.fc1 = nn.Linear(in_features=512*H_out*W_out, out_features=1024)
		self.fc2 = nn.Linear(in_features=1024, out_features=1)

	def forward(self, X):
		X = self.conv1(X)
		X = self.conv_blks(X)
		print(X.shape)
		X = X.flatten(start_dim=1)
		X = F.leaky_relu(self.fc1(X), negative_slope=0.2)
		return self.fc2(X)

	def _get_HW_out(self, H, W):
		for _ in range(4):
			H = math.ceil(H/2)
			W = math.ceil(W/2)
		return H, W

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, strides=1):
		super(ConvBlock, self).__init__()
		self.blk = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(negative_slope=0.2)
		)

	def forward(self, X):
		return self.blk(X)