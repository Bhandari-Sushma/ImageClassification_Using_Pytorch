# Imports
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from helper import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
epoch = 5


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load Pre-trained model and retrain it
model = torchvision.models.vgg16(pretrained=True)
model.avgpool = Identity()
model.classifier = nn.Linear(512, num_classes)
model.to(device)

# Load Data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Dataset.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = Dataset.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train(model=model, optimizer=optimizer, loss_function=criterion, dataloader=trainloader, epoch=epoch)
print("Checking accuracy on train data.")
check_accuracy(trainloader, model)
print("Checking accuracy in test data.")
check_accuracy(testloader, model)


#Save the model
PATH = '../Saved_Models/VGG16_cifar.pth'
torch.save(model, PATH)