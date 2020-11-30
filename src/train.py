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
	### Data
	hr_train_loader, lr_train_loader, hr_val_loader, lr_val_loader = load_data()

	### Model
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE)
	G.apply(xavier_init_weights).to(device)
	D = Discriminator()
	D.apply(xavier_init_weights).to(device)
	summary(G, input_size=(3, LR_CROPPED_SIZE, LR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
	summary(D, input_size=(3, HR_CROPPED_SIZE, HR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))

	### Training
	criterion = torch.nn.BCELoss()
	optimizerG = optim.Adam(G.parameters())
	optimizerD = optim.Adam(D.parameters())

	real_value = 1.0
	fake_value = 0.0

	for epoch in range(EPOCHS):
		print(f"\nEpoch: {epoch}")

		for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
			print(f"\tBatch: {batch}/{len(hr_train_loader)//BATCH_SIZE}")

			# Transfer data to GPU if available
			hr_batch, lr_batch = hr_batch.to(device), lr_batch.to(device)

			#### TRAIN D: maximize `log(D(x)) + log(1-D(G(z)))`
			optimizerD.zero_grad()

			# Classify all-real HR images
			real_labels = torch.full(size=(len(hr_batch),), fill_value=real_value, dtype=torch.float, device=device)
			output_real = D(hr_batch).view(-1)
			err_real = criterion(output_real, real_labels)
			err_real.backward()

			# Classify all-fake HR images (or SR images)
			fake_labels = torch.full(size=(len(hr_batch),), fill_value=fake_value, dtype=torch.float, device=device)
			sr_img = G(lr_batch)
			output_fake = D(sr_img.detach()).view(-1)
			err_fake = criterion(output_fake, fake_labels)
			err_fake.backward()

			optimizerD.step()

			# Logging
			D_x = output_real.mean().item()
			D_Gz1 = output_fake.mean().item()
			err_D = err_real + err_fake

			#### TRAIN G: minimize `log(D(G(z))`
			optimizerG.zero_grad()

			output_fake = D(sr_img).view(-1)
			err_G = criterion(output_fake, real_labels)
			err_G.backward()

			optimizerG.step()

			# Logging stats
			D_Gz2 = output_fake.mean().item()
			print(f"   err_D: {err_D.item():.4f}; err_G: {err_G.item():.4f}; D_x: {D_x:.4f}; "
				  f"D_Gz1: {D_Gz1:.4f}; D_Gz2: {D_Gz2:.4f}")

			## Free up GPU memory
			del hr_batch, lr_batch, err_D, err_G, real_labels, fake_labels, output_real, output_fake, sr_img
			torch.cuda.empty_cache()

def load_data():
	data_train_hr = DIV2K(data_dir=TRAIN_HR_DIR, transform=CenterCrop(size=HR_CROPPED_SIZE))
	data_train_lr = DIV2K(data_dir=TRAIN_LR_DIR, transform=CenterCrop(size=LR_CROPPED_SIZE))
	data_val_hr = DIV2K(data_dir=VAL_HR_DIR, transform=CenterCrop(size=HR_CROPPED_SIZE))
	data_val_lr = DIV2K(data_dir=VAL_LR_DIR, transform=CenterCrop(size=LR_CROPPED_SIZE))

	hr_train_loader = DataLoader(dataset=data_train_hr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	lr_train_loader = DataLoader(dataset=data_train_lr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	hr_val_loader = DataLoader(dataset=data_val_hr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
	lr_val_loader = DataLoader(dataset=data_val_lr, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)

	assert len(hr_train_loader) == len(lr_train_loader) and len(hr_val_loader) == len(lr_val_loader)

	return hr_train_loader, lr_train_loader, hr_val_loader, lr_val_loader

def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)

if __name__ == "__main__":
	train()