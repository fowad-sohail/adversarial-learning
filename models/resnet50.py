import torch
import torch.nn as nn
import torch.nn.functional as func

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(1, './utils/')
from load_and_test import load_cifar10, train, test, adversarial_test


train_loader, test_loader = load_cifar10()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.to(device)

train(model, device, train_loader)
test(model, device, test_loader)

accuracies = []
examples = []
epsilons = [0, .05, .1, .15, .2, .25, .3]


# Run test for each epsilon
for eps in epsilons:
    # FGSM
    acc, ex = adversarial_test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
    # CW attack
    cw_acc, cw_ex = adversarial_test_cw(model, device, test_loader, epsilon)
    accuracies.append(acc)
    examples.append(ex)
    cw_acc, cw_ex = adversarial_test_pgd(model, device, test_loader, epsilon)
    accuracies.append(acc)
    examples.append(ex)

