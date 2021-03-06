import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cw

def load_cifar10():
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='/data/pytorch/', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='/data/pytorch/', train=False,
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=0)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, testloader



def train(model, device, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

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

def test(model, device, test_loader):
    criterion = nn.CrossEntropyLoss()
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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def adversarial_test(model, device, test_loader, epsilon ):
    model.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.argmax(1, keepdim=True)[1] # get the index of the max log-probability
        

        fixed_init_pred = init_pred.cpu()
        fixed_target = target.argmax(0, keepdim=True).cpu()

        # If the initial prediction is wrong, dont bother attacking, just move on
        if fixed_init_pred.item() != fixed_target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.argmax(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.cpu().item() == fixed_target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (fixed_init_pred.item(), final_pred.cpu().item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (fixed_init_pred.item(), final_pred.cpu().item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))*100
    print("Epsilon: {}\tTest Accuracy = {} / {} = {} %".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def cw_attack(model, dataloader, mean, std):
    inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
                max((1 - m) / s for m, s in zip(mean, std)))
    # an untargeted adversary
    adversary = cw.L2Adversary(targeted=False,
                            confidence=0.0,
                            search_steps=10,
                            box=inputs_box,
                            optimizer_lr=5e-4)

    inputs, targets = next(iter(dataloader))
    adversarial_examples = adversary(model, inputs, targets, to_numpy=False)

    # a targeted adversary
    adversary = cw.L2Adversary(targeted=True,
                            confidence=0.0,
                            search_steps=10,
                            box=inputs_box,
                            optimizer_lr=5e-4)

    inputs, _ = next(iter(dataloader))
    # a batch of any attack targets
    attack_targets = torch.ones(inputs.size(0)) * 3
    adversarial_examples = adversary(net, inputs, attack_targets, to_numpy=False)


def adversarial_test_cw(model, device, test_loader, epsilon):
    model.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.argmax(1, keepdim=True)[1] # get the index of the max log-probability
        

        fixed_init_pred = init_pred.cpu()
        fixed_target = target.argmax(0, keepdim=True).cpu()

        # If the initial prediction is wrong, dont bother attacking, just move on
        if fixed_init_pred.item() != fixed_target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call CW Attack
        perturbed_data = cw_attack(model, dataloader, mean, std)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.argmax(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.cpu().item() == fixed_target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (fixed_init_pred.item(), final_pred.cpu().item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (fixed_init_pred.item(), final_pred.cpu().item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))*100
    print(final_acc)

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def pgd_attack(model, eps=8/255, alpha=2/255, steps=4, test_loader, labels):
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
    adversarial_images = atk(test_loader, labels)
    return adversarial_images

def adversarial_test_pgd(model, device, test_loader, epsilon):
    model.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.argmax(1, keepdim=True)[1] # get the index of the max log-probability
        

        fixed_init_pred = init_pred.cpu()
        fixed_target = target.argmax(0, keepdim=True).cpu()

        # If the initial prediction is wrong, dont bother attacking, just move on
        if fixed_init_pred.item() != fixed_target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call PGD Attack
        perturbed_data = pgd_attack(model, test_loader)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.argmax(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.cpu().item() == fixed_target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (fixed_init_pred.item(), final_pred.cpu().item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (fixed_init_pred.item(), final_pred.cpu().item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))*100
    print(final_acc)

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples