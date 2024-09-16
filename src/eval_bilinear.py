import sys, os
import numpy as np
from glob import glob
import dippykit as dip
from dippykit.sampling import resample
from dippykit.metrics import PSNR, SSIM

sys.path.append('/content/gdrive/MyDrive/DIP/')
from CONFIG import VAL_HR_DIR, VAL_LR_DIR, UPSCALE

INTERPOLATION = 'bilinear'

def eval_bilinear():
	M = np.array([[1.0/UPSCALE, 0], 
					[0, 1.0/UPSCALE]])

	psnr_result = []

	lr_imgs = glob(f"{VAL_LR_DIR}/*.png")
	for lr_path in lr_imgs:
		file_name = lr_path.split('/')[-1]
		file_name = file_name.replace("x4", '').replace("x2", '')
		hr_path = os.path.join(VAL_HR_DIR, file_name)

		lr = dip.im_to_float(dip.im_read(lr_path))
		hr = dip.im_to_float(dip.im_read(hr_path))

		# Upsample LR img using bilinear
		y = resample(lr*255, M, interpolation=INTERPOLATION)
		y = y/255.0
		h, w = y.shape[:2]
		hr = hr[:h, :w, :]
		
		psnr_ = PSNR(y, hr, max_signal_value=1.0)
		ssim_, _ = SSIM(y, hr, multichannel=True)
		psnr_result.append(psnr_)

		print("psnr", file_name, psnr_)
		print("ssim", file_name, ssim_)

	print("\nMean PSNR:", np.mean(psnr_result))
	print("STD PSNR:", np.std(psnr_result))
	print("Min PSNR:", np.min(psnr_result))
	print("Max PSNR:", np.max(psnr_result))

	print("\nMean SSIM:", np.mean(ssim_result))
	print("STD SSIM:", np.std(ssim_result))
	print("Min SSIM:", np.min(ssim_result))
	print("Max SSIM:", np.max(ssim_result))


if __name__ == "__main__":
	eval_bilinear()