import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader
from dataset import DIV2K
from generator import Generator
from discriminator import Discriminator
from train import PATH_G, PATH_D
from src.CONFIG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {str(device).upper()}")

def load_test_data():
	data_test_hr = DIV2K(data_dir=VAL_HR_DIR, transform=CenterCrop(size=HR_CROPPED_SIZE))
	data_test_lr = DIV2K(data_dir=VAL_LR_DIR, transform=CenterCrop(size=LR_CROPPED_SIZE))
	hr_test_loader = DataLoader(dataset=data_test_hr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	lr_test_loader = DataLoader(dataset=data_test_lr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)

	assert len(hr_test_loader) == len(lr_test_loader)
	return hr_test_loader, lr_test_loader
