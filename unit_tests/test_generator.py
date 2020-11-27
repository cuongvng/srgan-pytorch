import unittest
import sys
import torch
sys.path.append("../")
from src.gan import Residual_Block, PixelShuffler, Generator

class TestGenerator(unittest.TestCase):
	def test_res_blk(self):
		X = torch.randn(size=(2, 3, 100, 100))
		res_blk = Residual_Block(in_channels=3, out_channels=16, strides=1)
		self.assertEqual(res_blk(X).shape, (2, 16, 100, 100))

if __name__ == '__main__':
	unittest.main()

