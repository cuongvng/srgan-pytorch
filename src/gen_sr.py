import torch
import torchvision.transforms as trf
from PIL import Image
from generator import Generator
from train import PATH_G, xavier_init_weights, transform_lr
import os
import sys
sys.path.append('../')
from CONFIG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR_DIR = "../sr_img"

def generate_sr(lr_img_path):
	with torch.no_grad():
		pil_img = Image.open(lr_img_path)
		img_tensor = trf.ToTensor()(pil_img)
		img_tensor = torch.unsqueeze(img_tensor, 0) # add batch dimension
		sr_img = G(img_tensor)
		print(f"Upscaled from size [{img_tensor.shape[2]}, {img_tensor.shape[3]}] to [{sr_img.shape[2]}, {sr_img.shape[3]}]")

	file_name = lr_img_path.split('/')[-1]
	sr_img_path = os.path.join(SR_DIR, f"sr_{file_name}")
	tensor_to_img(sr_img, sr_img_path)

def tensor_to_img(tensor, filepath):
	pil = trf.ToPILImage()(tensor.squeeze_(0))
	pil.save(filepath)
	print(f"Saved to {filepath}")

if __name__ == '__main__':
	try:
		os.mkdir(SR_DIR)
	except FileExistsError:
		pass

	# Load checkpoints
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=UPSCALE)
	if PATH_G.exists():
		checkpoint_G = torch.load(PATH_G)
		G.load_state_dict(checkpoint_G['state_dict']).to(device)
	else:
		print("Checkpoints not found, using Xavier initialization.")
		G.apply(xavier_init_weights).to(device)
	G.eval()

	generate_sr("../DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png")