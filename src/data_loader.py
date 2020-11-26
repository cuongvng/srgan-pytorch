import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np

class DIV2K(Dataset):
	def __init__(self, data_dir):
		# Get all paths of images inside `data_dir` into a list
		pattern = os.path.join(data_dir, "*.png")
		self.file_paths = glob.glob(pattern, recursive=True)

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):
		img_tensor = self.load_img(self.file_paths[index])
		return img_tensor

	def load_img(self, file_path):
		img = Image.open(file_path)
		img_data = np.array(img.getdata())
		return torch.tensor(img_data, dtype=torch.float32)