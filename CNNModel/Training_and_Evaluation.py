# Imports
import torch
import torch.nn as nn
from CNNModel.model import ImageClassifierOnCIFAR
import torch.optim as optim
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from helper import *


# Set the device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Hyper-parameters
in_channels = 3
no_class = 10
learning_rate = 0.001
batch_size = 4
epoch = 3

# Load Data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Dataset.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = Dataset.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


# Initialize the network

model = ImageClassifierOnCIFAR(in_channel=3, num_class=no_class).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model=model, optimizer=optimizer, loss_function=criterion, dataloader=trainloader, epoch=epoch)
print("Checking accuracy on train data.")
check_accuracy(trainloader, model)
print("Checking accuracy in test data.")
check_accuracy(testloader, model)

#Save the model
PATH = '../Saved_Models/cifar_net.pth'
torch.save(model.state_dict(), PATH)

