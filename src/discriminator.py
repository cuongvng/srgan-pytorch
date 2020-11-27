import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		pass

	def forward(self, X):
		pass

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