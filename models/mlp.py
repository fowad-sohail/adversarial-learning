import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.insert(1, './utils/')

from load_and_test import load_cifar10, train, test, adversarial_test


train_loader, test_loader = load_cifar10()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,10)
        self.droput = nn.Dropout(0.2)
        
    def forward(self,x):
        x = x.view(-1,3*32*32)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = F.relu(self.fc3(x))
        x = self.droput(x)
        x = self.fc4(x)
        return x


model = MLP().to(device)

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

