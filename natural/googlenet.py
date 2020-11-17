import torch
import torch.nn as nn
import torch.nn.functional as func

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(1, './utils/')

from load_and_test import load_cifar10, train, test

train_loader, test_loader = load_cifar10()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
model.to(device)

train(model, device, train_loader)
test(model, device, test_loader)
