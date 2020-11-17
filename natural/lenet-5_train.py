import torch

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(1, '../models/')
sys.path.insert(1, '../utils')

from data_loader import load_cifar10
from train_test import train_test
from lenet5_model import LeNet

train_loader, test_loader = load_cifar10()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

model = LeNet().to(device)

train_test(model, device, train_loader, test_loader)