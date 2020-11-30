import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np

class DIV2K(Dataset):
	def __init__(self, data_dir, transform=None):
		# Get all paths of images inside `data_dir` into a list
		pattern = os.path.join(data_dir, "**/*.png")
		self.file_paths = glob.glob(pattern, recursive=True)
		self.transform = transform

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):
		img = Image.open(self.file_paths[index])

		if self.transform is not None:
			img = self.transform(img)

		img_data = np.array(img.getdata()).reshape((3, img.size[0], img.size[1]))
		img_tensor = torch.tensor(img_data, dtype=torch.float32)
		file_name = self.file_paths[index].split('/')[-1]
		return img_tensor, file_name