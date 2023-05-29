import torch
from piq import ssim, psnr
import sys, os
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob

sys.path.append('/content/gdrive/MyDrive/DIP/')
from gen_sr import SR_VAL_DIR
from CONFIG import VAL_HR_DIR

def load_img_to_tensor(filepath):
	img = Image.open(filepath)
	return ToTensor()(img)

def eval_PSNR_SSIM():
	"""
	Generate Super Resolution images and calculate PSNR
	"""

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

if __name__ == "__main__":
	eval_PSNR_SSIM()