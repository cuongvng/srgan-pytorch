import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np

class DIV2K(Dataset):
	def __init__(self, data_dir):
		# Get all paths of images inside `data_dir` into a list
		pattern = os.path.join(data_dir, "**/*.png")
		self.file_paths = glob.glob(pattern, recursive=True)

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):
		img_tensor = self.load_img(self.file_paths[index])
		return img_tensor

	def load_img(self, file_path):
		img = Image.open(file_path)
		img_data = np.array(img.getdata()).reshape((3, img.size[0], img.size[1]))
		return torch.tensor(img_data, dtype=torch.float32)

def test():
	data = DIV2K(data_dir="../DIV2K/DIV2K_train_LR_bicubic/X4")
	print(data[3].shape)
	print(data[5].shape)

if __name__ == "__main__":
	test()