import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary
import sys
sys.path.append("../")
from dataset import DIV2K
from generator import Generator
from discriminator import Discriminator
from src.CONFIG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {str(device).upper()}")

def train():
	# Data
	hr_train_loader, lr_train_loader, hr_val_loader, lr_val_loader = load_data()

	# Model
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE).apply(xavier_init_weights).to(device)
	D = Discriminator().apply(xavier_init_weights).to(device)
	summary(G, input_size=(3, LR_CROPPED_SIZE, LR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
	summary(D, input_size=(3, HR_CROPPED_SIZE, HR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))


	# Training

def load_data():
	data_train_hr = DIV2K(data_dir=TRAIN_HR_DIR, transform=CenterCrop(size=HR_CROPPED_SIZE))
	data_train_lr = DIV2K(data_dir=TRAIN_LR_DIR, transform=CenterCrop(size=LR_CROPPED_SIZE))
	data_val_hr = DIV2K(data_dir=VAL_HR_DIR, transform=CenterCrop(size=HR_CROPPED_SIZE))
	data_val_lr = DIV2K(data_dir=VAL_LR_DIR, transform=CenterCrop(size=LR_CROPPED_SIZE))

	hr_train_loader = DataLoader(dataset=data_train_hr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	lr_train_loader = DataLoader(dataset=data_train_lr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	hr_val_loader = DataLoader(dataset=data_val_hr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	lr_val_loader = DataLoader(dataset=data_val_lr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)

	return hr_train_loader, lr_train_loader, hr_val_loader, lr_val_loader

def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)

if __name__ == "__main__":
	train()