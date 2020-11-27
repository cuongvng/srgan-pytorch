import unittest
import sys
sys.path.append("../")
import torch
from src.discriminator import Discriminator, ConvBlock

class TestDiscriminator(unittest.TestCase):
	def test_conv_blk(self):
		N, C_in, H, W = 2, 16, 100, 100
		C_out = 2*C_in
		X = torch.randn(size=(N, C_in, H, W))
		conv1 = ConvBlock(in_channels=C_in, out_channels=C_in, strides=1)
		conv2 = ConvBlock(in_channels=C_in, out_channels=C_out, strides=2)
		out1, out2 = conv1(X), conv2(X)
		self.assertEqual(out1.shape, (N, C_in, H, W))
		self.assertEqual(out2.shape, (N, C_out, H//2, W//2))

if __name__ == '__main__':
	unittest.main()