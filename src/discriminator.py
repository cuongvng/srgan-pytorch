import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")

class Discriminator(nn.Module):
	def __init__(self):
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
		self.global_pooling = nn.AdaptiveAvgPool2d(output_size=1)
		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.conv3 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1)

	def forward(self, X):
		X = self.conv1(X)
		X = self.conv_blks(X)
		X = self.global_pooling(X)
		X = self.conv2(X)
		X = self.conv3(X)
		X = X.flatten(start_dim=1)
		return F.sigmoid(X)

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