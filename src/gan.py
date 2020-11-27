import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, n_blocks, upscale = 4):
		super(Generator, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
		self.prelu1 =  nn.PReLU()

		self.res_blocks = nn.Sequential()
		for i in range(n_blocks):
			self.res_blocks.add_module(f"res_blk{i}",
									   Residual_Block(in_channels=64, out_channels=64,strides=1, use_1x1_conv=False))
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(64)

		self.pixel_shufflers = nn.Sequential()
		for i in range(2):
			pass


	def forward(self, X):
		pass

class Residual_Block(nn.Module):
	def __init__(self, in_channels, out_channels, strides, use_1x1_conv=True):
		super(Residual_Block, self).__init__()

		self.use_1x1_conv = use_1x1_conv
		self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
		self.blk = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
			nn.BatchNorm2d(out_channels),
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

class PixelShuffler(nn.Module):
	def __init__(self):
		super(PixelShuffler, self).__init__()

	def forward(self, X):
		pass