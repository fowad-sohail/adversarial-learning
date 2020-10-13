import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


input_size = 3*32*32
output_size = 10
batch_size = 128


# Generic base model structure for classification problem...extend-able to any problem...does not contain model architecture (i.e. no __init__ and __forward__ methods)
# built on top of built-in torch.nn.Module class of PyTorch
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch              
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # calculate loss through cross-entropy having negative log likelihood as well as softmax for classification task.
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs] # extract loss from a batch
        epoch_loss = torch.stack(batch_losses).mean()   # combine loss for an epoch

        batch_accs = [x['val_acc'] for x in outputs]    # extract accuracy of a batch
        epoch_acc = torch.stack(batch_accs).mean()      # combine accuracy for an epoch
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()} # .item() gives (key, value) pairs from dictionary as tuples
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
# The architecture of the model
class CIFAR10Model(ImageClassificationBase):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(input_size, 512)
    self.bn1 = nn.BatchNorm1d(num_features=512)
    self.fc2 = nn.Linear(512, 512)
    self.bn2 = nn.BatchNorm1d(num_features=512)
    self.fc3 = nn.Linear(512, 512)
    self.bn3 = nn.BatchNorm1d(num_features=512)
    self.fc4 = nn.Linear(512, 512)
    self.bn4 = nn.BatchNorm1d(num_features=512)
    self.fc5 = nn.Linear(512, 512)
    self.bn5 = nn.BatchNorm1d(num_features=512)
    self.linear6 = nn.Linear(512, output_size)
  
  def forward(self, xb):
    # Flatten the input images....as vectors
    out = xb.view(xb.size(0), -1)

    out = self.fc1(out)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.fc2(out)
    out = self.bn2(out)
    out = F.relu(out)

    out = self.fc3(out)
    out = self.bn3(out)
    out = F.relu(out)

    out = self.fc4(out)
    out = self.bn4(out)
    out = F.relu(out)

    out = self.fc5(out)
    out = self.bn5(out)
    out = F.relu(out)

    out = self.linear6(out)
    out = F.softmax(out, dim=1)
    
    return out

def load_model():
    model = CIFAR10Model()
    model = torch.load('./8Oct_model_cifar10_5layer.pth')
    model = model.eval()
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()
    return model


# Downloads data to ./data directory
def load_cifar10():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2,
                                            shuffle=False, num_workers=4)
    return test_set, test_loader

# From Asim's colab
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# From pytorch docs
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    
    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    model = load_model()

    # Test on normal CIFAR10
    test_set, test_loader = load_cifar10()
    print('\nTESTING ON NORMAL CIFAR10')
    test(model, 'cuda', test_loader)

    # Test on adversarial CIFAR10
    print('\nTESTING ON ADVERSARIAL CIFAR10')
    ART_classifier = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), input_shape=input_size, nb_classes=10)
    attack = FastGradientMethod(estimator=ART_classifier, eps=0.2)
    adversarial_loader = attack.generate(x=test_set)
    test(model, 'cuda', adversarial_loader)
