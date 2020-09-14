# Imports
import torch

# Set the device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, optimizer, loss_function, dataloader, epoch):
    """ Train the model

        Args:
            model : (torch.nn.Module) the neural network
            optimizer : (torch.optim) optimizer for parameters of model
            loss_function : function that takes batch_output and batch_target and compute the loss for the batch
            dataloader : (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            epoch : number of ephoes for training the network
        """

    # set model to training mode
    model.train()
    print("\n------Training Started-----")
    print("\n----- Number of Epoches : ", epoch)

    for each_epoch in range(epoch):
        print("\nEpoch : ", each_epoch+1)
        for batch_idx, data in enumerate(dataloader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            # zero the parameters gradients
            optimizer.zero_grad()

            # forward
            output = model(images)
            loss = loss_function(output, labels)

            # backward
            loss.backward()

            # update
            optimizer.step()


def check_accuracy(dataloader, model):
    """ Check the accuracy of the model (better to use after training)

        Args:
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training or test data
            model: (torch.nn.Module) the neural network
    """
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, target = data

            img = img.to(device)
            target = target.to(device)

            scores = model(img)
            _, prediction = scores.max(1)
            total += target.size(0)
            correct += (prediction == target).sum()

        print(f"\n{correct} / {total} were correct, which means the accuracy is : {float(correct) / float(total) * 100 :.2f}")
