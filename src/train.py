import torch
import torchvision.transforms as trf
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary
from pathlib import Path
import sys
sys.path.append("../")
import os
from dataset import DIV2K
from generator import Generator
from discriminator import Discriminator
from CONFIG import *
from loss import PerceptualLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {str(device).upper()}")

try:
	os.mkdir("../model")
except FileExistsError:
	pass

PATH_G = Path('../model/G.pt')
PATH_D = Path('../model/D.pt')

transform_hr = trf.Compose([
	trf.CenterCrop(HR_CROPPED_SIZE),
	trf.ToTensor()
])
transform_lr = trf.Compose([
	trf.CenterCrop(LR_CROPPED_SIZE),
	trf.ToTensor()
])

def train(resume_training=True):
	'''
	:param `resume_training`: whether to continue training from previous checkpoint or not.
	If checkpoints cannot be found, train from beginning, regardless of `resume_training`.
	'''
	### Load data
	data_train_hr, data_train_lr = load_training_data()
	hr_train_loader = DataLoader(dataset=data_train_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
	lr_train_loader = DataLoader(dataset=data_train_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
	assert len(hr_train_loader) == len(lr_train_loader)

	### Load models
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE).to(device)
	D = Discriminator().to(device)
	optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
	optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=BETAS)

	if resume_training and PATH_G.exists() and PATH_D.exists():
		G, D, optimizer_G, optimizer_D, prev_epochs = load_checkpoints(G, D, optimizer_G, optimizer_D)
		print("Continue training from previous checkpoints ...")
	else:
		G.apply(xavier_init_weights)
		D.apply(xavier_init_weights)
		prev_epochs = 0
		summary(G, input_size=(3, LR_CROPPED_SIZE, LR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
		summary(D, input_size=(3, HR_CROPPED_SIZE, HR_CROPPED_SIZE), batch_size=BATCH_SIZE, device=str(device))
		print("Training from start ...")

	### Train
	G.train()
	D.train()

	criterion_G = PerceptualLoss(vgg_coef=VGG_LOSS_COEF, adversarial_coef=ADVERSARIAL_LOSS_COEF).to(device)
	criterion_D = torch.nn.BCELoss()

	## Warm up G
	for w in range(5):
		print(f"\nWarmup: {w+1}")
		for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
			hr_img, lr_img = hr_batch[0].to(device), lr_batch[0].to(device)
			optimizer_G.zero_grad()

			sr_img = G(lr_img)
			output_fake = D(sr_img).view(-1)
			err_G = criterion_G(sr_img, hr_img, output_fake)
			err_G.backward()

			optimizer_G.step()
			if batch % 10 == 0:
				print(f"\tBatch: {batch + 1}/{len(data_train_hr) // BATCH_SIZE}")
				print(f"\terr_G: {err_G.item():.4f}")

	for e in range(EPOCHS):
		print(f"\nEpoch: {e+prev_epochs+1}")

		for (batch, hr_batch), lr_batch in zip(enumerate(hr_train_loader), lr_train_loader):
			# Transfer data to GPU if available
			hr_img, lr_img = hr_batch[0].to(device), lr_batch[0].to(device)

			#### TRAIN D: maximize `log(D(x)) + log(1-D(G(z)))`
			optimizer_D.zero_grad()

			# Classify all-real HR images
			real_labels = torch.full(size=(len(hr_img),), fill_value=REAL_VALUE, dtype=torch.float, device=device)
			output_real = D(hr_img).view(-1)
			err_D_real = criterion_D(output_real, real_labels)
			err_D_real.backward()

			# Classify all-fake HR images (or SR images)
			fake_labels = torch.full(size=(len(hr_img),), fill_value=FAKE_VALUE, dtype=torch.float, device=device)
			sr_img = G(lr_img)
			output_fake = D(sr_img.detach()).view(-1)
			err_D_fake = criterion_D(output_fake, fake_labels)
			err_D_fake.backward()

			optimizer_D.step()
			D_Gz1 = output_fake.mean().item()

			#### TRAIN G: minimize `log(D(G(z))`
			optimizer_G.zero_grad()

			output_fake = D(sr_img).view(-1)
			err_G = criterion_G(sr_img, hr_img, output_fake)
			err_G.backward()

			optimizer_G.step()

			# Print stats
			if batch%10==0:
				print(f"\tBatch: {batch + 1}/{len(data_train_hr) // BATCH_SIZE}")
				D_x = output_real.mean().item()
				D_Gz2 = output_fake.mean().item()
				print(f"\terr_D_real: {err_D_real.item():.4f}; err_D_fake: {err_D_fake.item():.4f}; "
					  f" err_G: {err_G.item():.4f}; D_x: {D_x:.4f}; D_Gz1: {D_Gz1:.4f}; D_Gz2: {D_Gz2:.4f}")

			## Free up GPU memory
			del hr_img, lr_img, err_D_fake, err_D_real, err_G, real_labels, fake_labels, output_real, output_fake, sr_img
			torch.cuda.empty_cache()

		### Save checkpoints
		save_checkpoints(G, D, optimizer_G, optimizer_D, epoch=prev_epochs+e+1)

def load_training_data():
	data_train_hr = DIV2K(data_dir=os.path.join("../", TRAIN_HR_DIR), transform=transform_hr)
	data_train_lr = DIV2K(data_dir=os.path.join("../", TRAIN_LR_DIR), transform=transform_lr)
	return data_train_hr, data_train_lr

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

def load_checkpoints(G, D, optimizerG, optimizerD):
	print("Loading checkpoints ...")
	checkpoint_G = torch.load(PATH_G)
	checkpoint_D = torch.load(PATH_D)
	G.load_state_dict(checkpoint_G['state_dict'])
	optimizerG.load_state_dict(checkpoint_G['optimizer'])
	D.load_state_dict(checkpoint_D['state_dict'])
	optimizerD.load_state_dict(checkpoint_D['optimizer'])
	prev_epochs = checkpoint_G['epoch']

	print("Loaded checkpoints successfully!")
	return G, D, optimizerG, optimizerD, prev_epochs

def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)

if __name__ == "__main__":
	train(resume_training=True)