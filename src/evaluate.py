import torch
import torchvision.transforms as trf
from torch.utils.data import DataLoader
from dataset import DIV2K
from generator import Generator
from discriminator import Discriminator
from train import PATH_G, PATH_D, xavier_init_weights, transform_lr, transform_hr
import os
import sys
sys.path.append('../')
from CONFIG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {str(device).upper()}")

def run_eval():
	"""
	Generate Super Resolution images and calculate losses of G and D
	"""
	# Load data
	hr_test_loader, lr_test_loader = load_test_data()

	# Load checkpoints
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE)
	D = Discriminator()

	if PATH_G.exists() and PATH_D.exists():
		G, D = load_checkpoints(G, D)
		G.to(device)
		D.to(device)
	else:
		print("Checkpoints not found!")
		return

	# Eval mode
	G.eval()
	D.eval()
	criterion = torch.nn.BCELoss()

	errors_D = []
	errors_G = []

	with torch.no_grad():
		for (i, hr_batch), lr_batch in zip(enumerate(hr_test_loader), lr_test_loader):
			hr_img, hr_names = hr_batch[0].to(device), hr_batch[1]
			lr_img, lr_names = lr_batch[0].to(device), lr_batch[1]

			# Eval D
			real_labels = torch.full(size=(len(hr_img),), fill_value=REAL_VALUE, dtype=torch.float, device=device)
			output_real = D(hr_img)
			err_D_real = criterion(output_real, real_labels)

			fake_labels = torch.full(size=(len(hr_img),), fill_value=FAKE_VALUE, dtype=torch.float, device=device)
			sr_img = G(lr_img)
			output_fake = D(sr_img)
			err_D_fake = criterion(output_fake, fake_labels)

			errors_D.append(err_D_real.item() + err_D_fake.item())

			# Eval G
			err_G = criterion(output_fake, real_labels)
			errors_G.append(err_G.item())

		print(f"\terr_D: {sum(errors_D)/len(errors_D):.4f}; err_G: {sum(errors_G)/len(errors_G):.4f}")

def load_test_data():
	data_test_hr = DIV2K(data_dir=os.path.join("../", VAL_HR_DIR), transform=transform_hr)
	data_test_lr = DIV2K(data_dir=os.path.join("../", VAL_LR_DIR), transform=transform_lr)
	hr_test_loader = DataLoader(dataset=data_test_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
	lr_test_loader = DataLoader(dataset=data_test_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)

	assert len(hr_test_loader) == len(lr_test_loader)
	return hr_test_loader, lr_test_loader

def load_checkpoints(G, D):
	checkpoint_G = torch.load(PATH_G)
	checkpoint_D = torch.load(PATH_D)
	G.load_state_dict(checkpoint_G['state_dict']).to(device)
	D.load_state_dict(checkpoint_D['state_dict']).to(device)

	print("Loaded checkpoints successfully!")
	return G, D