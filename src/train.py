import torch
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary
from pathlib import Path
import sys
sys.path.append("../")
from dataset import DIV2K
from generator import Generator
from discriminator import Discriminator
from src.CONFIG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {str(device).upper()}")

PATH_G = Path('../model/G.pt')
PATH_D = Path('../model/D.pt')

def train(resume_training=True):
	'''
	:param `resume_training`: whether to continue training from previous checkpoint or not.
	If checkpoints cannot be found, train from beginning, regardless of `resume_training`.
	'''
	### Load data
	hr_train_loader, lr_train_loader, hr_val_loader, lr_val_loader = load_data()

	### Load models
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE)
	D = Discriminator()
	optimizerG = optim.Adam(G.parameters())
	optimizerD = optim.Adam(D.parameters())
	G, D, optimizerG, optimizerD, prev_epochs = load_checkpoints(resume_training, G, D, optimizerG, optimizerD)

	### Train
	G.train()
	D.train()

	criterion = torch.nn.BCELoss()
	real_value = 1.0
	fake_value = 0.0

	for e in range(EPOCHS):
		print(f"\nEpoch: {e+prev_epochs+1}")

		for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
			print(f"\tBatch: {batch}/{len(hr_train_loader)//BATCH_SIZE}")

			# Transfer data to GPU if available
			hr_batch, lr_batch = hr_batch.to(device), lr_batch.to(device)

			#### TRAIN D: maximize `log(D(x)) + log(1-D(G(z)))`
			optimizerD.zero_grad()

			# Classify all-real HR images
			real_labels = torch.full(size=(len(hr_batch),), fill_value=real_value, dtype=torch.float, device=device)
			output_real = D(hr_batch).view(-1)
			err_D_real = criterion(output_real, real_labels)
			err_D_real.backward()

			# Classify all-fake HR images (or SR images)
			fake_labels = torch.full(size=(len(hr_batch),), fill_value=fake_value, dtype=torch.float, device=device)
			sr_img = G(lr_batch)
			output_fake = D(sr_img.detach()).view(-1)
			err_D_fake = criterion(output_fake, fake_labels)
			err_D_fake.backward()

			optimizerD.step()

			# For logging
			D_x = output_real.mean().item()
			D_Gz1 = output_fake.mean().item()
			err_D = err_D_real + err_D_fake

			#### TRAIN G: minimize `log(D(G(z))`
			optimizerG.zero_grad()

			output_fake = D(sr_img).view(-1)
			err_G = criterion(output_fake, real_labels)
			err_G.backward()

			optimizerG.step()

			# Print stats
			D_Gz2 = output_fake.mean().item()
			print(f"   err_D: {err_D.item():.4f}; err_G: {err_G.item():.4f}; D_x: {D_x:.4f}; "
				  f"D_Gz1: {D_Gz1:.4f}; D_Gz2: {D_Gz2:.4f}")

			## Free up GPU memory
			del hr_batch, lr_batch, err_D, err_G, real_labels, fake_labels, output_real, output_fake, sr_img
			torch.cuda.empty_cache()

		### Save checkpoints
		save_checkpoints(G, D, optimizerG, optimizerD, epoch=prev_epochs+e+1)

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

def save_checkpoints(G, D, optimizer_G, optimizer_D, epoch):
	checkpoint_G = {
		'model': G,
		'state_dict': G.state_dict(),
		'optimizer': optimizer_G.state_dict(),
		'epoch': epoch
	}
	checkpoint_D = {
		'model': D,
		'state_dict': D.state_dict(),
		'optimizer': optimizer_D.state_dict(),
	}
	torch.save(checkpoint_G, PATH_G)
	torch.save(checkpoint_D, PATH_D)

def load_checkpoints(resume_training, G, D, optimizerG, optimizerD):
	if resume_training and PATH_G.exists() and PATH_D.exists():
		checkpoint_G = torch.load(PATH_G)
		checkpoint_D = torch.load(PATH_D)
		G.load_state_dict(checkpoint_G['state_dict']).to(device)
		optimizerG.load_state_dict(checkpoint_G['optimizer'])
		D.load_state_dict(checkpoint_D['state_dict']).to(device)
		optimizerD.load_state_dict(checkpoint_D['optimizer'])
		prev_epochs = checkpoint_G['epoch']
		summary(G, input_size=(3, LR_CROPPED_SIZE, LR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
		summary(D, input_size=(3, HR_CROPPED_SIZE, HR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
		print("Training from previous checkpoints ...")
	else:
		G.apply(xavier_init_weights).to(device)
		D.apply(xavier_init_weights).to(device)
		prev_epochs = 0
		summary(G, input_size=(3, LR_CROPPED_SIZE, LR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
		summary(D, input_size=(3, HR_CROPPED_SIZE, HR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
		print("Training from initial values ...")

	return G, D, optimizerG, optimizerD, prev_epochs

def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)

if __name__ == "__main__":
	train(resume_training=True)