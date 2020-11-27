import unittest
import sys
import torch
sys.path.append("../")
from src.generator import Residual_Block, PixelShufflerBlock, Generator

class TestGenerator(unittest.TestCase):
	def test_res_blk(self):
		N, C_in, H, W = 2, 3, 100, 100
		C_out = 16
		X = torch.randn(size=(N, C_in, H, W))
		res_blk = Residual_Block(in_channels=C_in, out_channels=C_out, strides=1)
		self.assertEqual(res_blk(X).shape, (N, C_out, H, W))

	def test_pixel_shuffle_blk(self):
		N, C, H, W = 7, 256, 40, 40
		upscale_factor = 2
		X = torch.rand(size=(N, C, H, W ))
		ps = PixelShufflerBlock(in_channels=C, upscale_factor=upscale_factor)
		out = ps(X)
		self.assertEqual(out.shape, (N, C/upscale_factor**2, H*upscale_factor, W*upscale_factor))

	def test_generator(self):
		N, C, H, W = 5, 3, 100, 100
		n_resblks, upscale_factor = 3, 4
		X = torch.rand(size=(N, C, H, W))
		g = Generator(n_resblks, upscale_factor)
		out = g(X)
		self.assertEqual(out.shape, (N, C, H*upscale_factor, W*upscale_factor))

if __name__ == '__main__':
	unittest.main()

