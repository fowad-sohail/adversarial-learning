import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def load_model():
    # PATH = 'path/to/saved/model'
    model = torch.load('./4Oct.pth')
    model = model.eval()
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model


def load_cifar10():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)



if __name__ == "__main__":
    # load_model()
    load_checkpoint('./4Oct.pth')
    # load_cifar10()
