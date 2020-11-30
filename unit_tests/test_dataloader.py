import unittest
import sys
sys.path.append("../")
import torchvision.transforms as trf
from src.dataset import DIV2K
from src.CONFIG import LR_CROPPED_SIZE, HR_CROPPED_SIZE

class TestDataLoader(unittest.TestCase):
	def test_train_lr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_train_LR_bicubic", transform=trf.CenterCrop(size=LR_CROPPED_SIZE))
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, LR_CROPPED_SIZE, LR_CROPPED_SIZE))

	def test_train_hr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_train_HR", transform=trf.CenterCrop(size=HR_CROPPED_SIZE))
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, HR_CROPPED_SIZE, HR_CROPPED_SIZE))

	def test_valid_lr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_valid_LR_bicubic", transform=trf.CenterCrop(size=LR_CROPPED_SIZE))
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, LR_CROPPED_SIZE, LR_CROPPED_SIZE))

	def test_valid_hr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_valid_HR", transform=trf.CenterCrop(size=HR_CROPPED_SIZE))
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, HR_CROPPED_SIZE, HR_CROPPED_SIZE))


if __name__ == '__main__':
	unittest.main()