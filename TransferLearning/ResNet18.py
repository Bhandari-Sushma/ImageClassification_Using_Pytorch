# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
in_channel = 3
mum_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epoch = 5

# Load Pre-trained model and retrain it

# Load Pre-trained model and retrain it
model = torchvision.models.resnet18(pretrained=True)
print(model)





# Load Data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Dataset.CIFAR10(root='dataset/', train=True, download=True, transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

testset = Dataset.CIFAR10(root='dataset/', train=False, download=True, transform=transform)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)