import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, n_res_blks, upscale_factor=4):
		super(Generator, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
		self.prelu1 = nn.PReLU()

		self.res_blocks = nn.Sequential()
		for i in range(n_res_blks):
			self.res_blocks.add_module(f"res_blk_{i}",
									   Residual_Block(in_channels=64, out_channels=64,strides=1, use_1x1_conv=False))
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(64)

		self.pixel_shufflers = nn.Sequential()
		for i in range(2):
			self.pixel_shufflers.add_module(f"pixel_shuffle_blk_{i}",
											PixelShufflerBlock(in_channels=64, upscale_factor=upscale_factor//2))
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4)

	def forward(self, X):
		X = self.prelu1(self.conv1(X))
		X_before_resblks = X.clone()

		X = self.res_blocks(X)
		X = self.bn(self.conv2(X))
		X = F.relu(X + X_before_resblks)

		X = self.pixel_shufflers(X)
		X = self.conv3(X)

		return F.tanh(X)

class Residual_Block(nn.Module):
	def __init__(self, in_channels, out_channels, strides, use_1x1_conv=True):
		super(Residual_Block, self).__init__()

		self.use_1x1_conv = use_1x1_conv
		self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
		self.blk = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
		)

	def forward(self, X):
		"""
		:param X: tensor with shape (N, C, H, W)
		"""
		X_original = X.clone()
		X = self.blk(X)
		if self.use_1x1_conv:
			X_original = self.conv1x1(X_original)

		return F.relu(X + X_original)

class PixelShufflerBlock(nn.Module):
	def __init__(self, in_channels, upscale_factor=2):
		super(PixelShufflerBlock, self).__init__()

		self.blk = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
			nn.PixelShuffle(upscale_factor=upscale_factor),
			nn.PReLU()
		)

	def forward(self, X):
		return self.blk(X)