import unittest
import sys
import torch
sys.path.append("../")
from src.gan import Residual_Block, PixelShufflerBlock, Generator

class TestGenerator(unittest.TestCase):
	def test_res_blk(self):
		N, C_in, H, W = 2, 3, 100, 100
		C_out = 16
		X = torch.randn(size=(N, C_in, H, W))
		res_blk = Residual_Block(in_channels=C_in, out_channels=C_out, strides=1)
		self.assertEqual(res_blk(X).shape, (N, C_out, H, W))

if __name__ == '__main__':
	unittest.main()

