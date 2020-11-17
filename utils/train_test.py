import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


def train_test(model, device, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss_list.append(loss.detach().cpu().numpy().item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.detach().cpu(), 1)        
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))

    print('Finished Training')
    train_acc = sum(acc_list) / len(acc_list)
    print('Training Accuracy: ' + str(train_acc))

    # Test the model
    val_loss_list = []
    val_acc_list = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            val_loss_list.append(loss.detach().cpu().numpy().item())
            _, predicted = torch.max(outputs.data.detach().cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc_list.append((correct / total) * 100)


        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

