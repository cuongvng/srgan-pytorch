import torch
import torch.optim as optim
import torchsummary
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
import sys
sys.path.append("../")
from data_loader import DIV2K
from CONFIG import LR_CROPPED_SIZE, HR_CROPPED_SIZE
from generator import Generator
from discriminator import Discriminator

def train():
	pass

def _load_data(data_dir):
	data_train_HR = DIV2K()

def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)
