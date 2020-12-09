import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(1, './utils/')
from load_and_test import load_cifar10, train, test, adversarial_test
from .model_xception import xception


train_loader, test_loader = load_cifar10()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

model = xception()
model.to(device)

train(model, device, train_loader)
test(model, device, test_loader)

accuracies = []
examples = []
epsilons = [0, .05, .1, .15, .2, .25, .3]

# Run test for each epsilon
for eps in epsilons:
    acc, ex = adversarial_test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

