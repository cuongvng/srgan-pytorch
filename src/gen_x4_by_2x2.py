import torch
import torchvision.transforms as trf
from piq import ssim, psnr
from PIL import Image
from generator import Generator
from pathlib import Path
import os
import glob
import sys
sys.path.append('/content/gdrive/MyDrive/DIP/')
from CONFIG import *
from pathlib import Path
from gen_sr import tensor_to_img
from evaluate import load_img_to_tensor
from glob import glob

X2_Generator = Path("/content/gdrive/MyDrive/DIP/model/G_x2.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR_VAL_DIR = f"/content/gdrive/MyDrive/DIP/results/sr_val_x4_by_2_x2"

def gen_4x_sr():
	# Load x2 checkpoints
	G = Generator(n_res_blks=N_RESBLK_G, upscale_factor=2)
	checkpoint_G = torch.load(X2_Generator)
	G.load_state_dict(checkpoint_G['state_dict'])
	G.to(device)
	G.eval()

	files = glob(f"/content/gdrive/MyDrive/DIP/DIV2K/DIV2K_valid_LR_bicubic/X4/*.png")
	for lr_img_path in files:
		with torch.no_grad():
			pil_img = Image.open(lr_img_path)
			img_tensor = trf.ToTensor()(pil_img)
			img_tensor = torch.unsqueeze(img_tensor, 0) # add batch dimension
			img_tensor = img_tensor.to(device)
			x2_img = G(img_tensor)
			x4_img = G(x2_img)
        
		file_name = lr_img_path.split('/')[-1]
		sr_img_path = os.path.join(SR_VAL_DIR, f"sr_{file_name}")
		tensor_to_img(x4_img, sr_img_path)

def eval_PSNR_SSIM():
	psnr_result = torch.tensor([])
	ssim_result = torch.tensor([])

	print("Load SR images from", SR_VAL_DIR)
	sr_vals = glob(f"{SR_VAL_DIR}/*.png")

	for sr_path in sr_vals:
		file_name = sr_path.split('/')[-1]
		file_name = file_name.replace("sr_", '').replace("x4", '').replace("x2", '')
		hr_path = os.path.join(VAL_HR_DIR, file_name)

		sr = load_img_to_tensor(sr_path)
		hr = load_img_to_tensor(hr_path)

		psnr_ = psnr(sr, hr, data_range=1.0, reduction='none')
		psnr_result = torch.cat((psnr_result, psnr_))
		print("psnr", file_name, psnr_)

		ssim_ = ssim(sr, hr, data_range=1.0, reduction='none')
		ssim_result = torch.cat((ssim_result, ssim_))
		print("ssim", file_name, ssim_)

	print("\nMean PSNR:", torch.mean(psnr_result))
	print("STD PSNR:", torch.std(psnr_result))
	print("Min PSNR:", torch.min(psnr_result))
	print("Max PSNR:", torch.max(psnr_result))

	print("\nMean SSIM:", torch.mean(ssim_result))
	print("STD SSIM:", torch.std(ssim_result))
	print("Min SSIM:", torch.min(ssim_result))
	print("Max SSIM:", torch.max(ssim_result))


if __name__ == '__main__':
	gen_4x_sr()
	eval_PSNR_SSIM()