import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import ToTensor

class DIV2K(Dataset):
	def __init__(self, data_dir, transform=ToTensor()):
		# Get all paths of images inside `data_dir` into a list
		pattern = os.path.join(data_dir, "**/*.png")
		self.file_paths = sorted(glob.glob(pattern, recursive=True))
		self.transform = transform

	def __len__(self):
		return len(self.file_paths)

	def __getitem__(self, index):
		file_name = self.file_paths[index].split('/')[-1]
		img = Image.open(self.file_paths[index])
		img = self.transform(img)
		return img, file_name