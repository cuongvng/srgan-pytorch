import unittest
import sys
sys.path.append("../")
import torch
import math
from src.discriminator import Discriminator, ConvBlock

class TestDiscriminator(unittest.TestCase):
	def test_conv_blk(self):
		N, C_in, H, W = 2, 16, 25, 25
		C_out = 2*C_in
		X = torch.randn(size=(N, C_in, H, W), requires_grad=False)
		conv1 = ConvBlock(in_channels=C_in, out_channels=C_in, strides=1)
		conv2 = ConvBlock(in_channels=C_in, out_channels=C_out, strides=2)
		with torch.no_grad():
			out1, out2 = conv1(X), conv2(X)
		self.assertEqual(out1.shape, (N, C_in, H, W))
		self.assertEqual(out2.shape, (N, C_out, math.ceil(H/2), math.ceil(W/2)))

	def test_discriminator(self):
		N, C_in, H, W = 2, 3, 223, 169
		X = torch.randn(size=(N, C_in, H, W), requires_grad=False)
		d = Discriminator(H, W)
		with torch.no_grad():
			out = d(X)
		self.assertEqual(out.shape, (N, 1))

if __name__ == '__main__':
	unittest.main()