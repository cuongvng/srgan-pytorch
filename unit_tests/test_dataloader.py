import unittest
import sys
sys.path.append("../")
import torchvision.transforms as trf
from src.dataset import DIV2K
from CONFIG import LR_CROPPED_SIZE, HR_CROPPED_SIZE

transform_hr = trf.Compose([
	trf.CenterCrop(HR_CROPPED_SIZE),
	trf.ToTensor()
])
transform_lr = trf.Compose([
	trf.CenterCrop(LR_CROPPED_SIZE),
	trf.ToTensor()
])

class TestDataLoader(unittest.TestCase):
	def test_train_lr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_train_LR_bicubic", transform=transform_lr)
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, LR_CROPPED_SIZE, LR_CROPPED_SIZE))

	def test_train_hr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_train_HR", transform=transform_hr)
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, HR_CROPPED_SIZE, HR_CROPPED_SIZE))

	def test_valid_lr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_valid_LR_bicubic", transform=transform_lr)
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, LR_CROPPED_SIZE, LR_CROPPED_SIZE))

	def test_valid_hr(self):
		data = DIV2K(data_dir="../DIV2K/DIV2K_valid_HR", transform=transform_hr)
		for i in range(len(data)):
			self.assertEqual(data[i][0].shape, (3, HR_CROPPED_SIZE, HR_CROPPED_SIZE))

	def test_matching_pairs(self):
		hr = DIV2K(data_dir="../DIV2K/DIV2K_train_HR", transform=trf.CenterCrop(size=HR_CROPPED_SIZE))
		lr = DIV2K(data_dir="../DIV2K/DIV2K_train_LR_bicubic", transform=trf.CenterCrop(size=LR_CROPPED_SIZE))
		for i in range(len(hr)):
			hr_name = hr[i][1]
			lr_name = lr[i][1]
			self.assertEqual(hr_name[0:4], lr_name[0:4])

if __name__ == '__main__':
	unittest.main()