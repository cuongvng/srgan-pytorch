import torch
import torch.nn as nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
	def __init__(self, vgg_coef, adversarial_coef):
		super(PerceptualLoss, self).__init__()
		_vgg19 = vgg19(pretrained=True)
		self.vgg19 = nn.Sequential(*_vgg19.features).eval()
		for p in self.vgg19.parameters():
			p.requires_grad = False
		self.euclidean_distance = nn.MSELoss()
		self.vgg_coef = vgg_coef
		self.adversarial_coef = adversarial_coef

	def forward(self, sr_img, hr_img, output_labels):
		adversarial_loss = torch.mean(1-output_labels)
		vgg_loss = self.euclidean_distance(self.vgg19(sr_img), self.vgg19(hr_img))
		pixel_loss = self.euclidean_distance(sr_img, hr_img)
		return pixel_loss, self.adversarial_coef*adversarial_loss, self.vgg_coef*vgg_loss