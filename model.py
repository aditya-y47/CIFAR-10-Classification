# Neural Network Structure defined as Net, taking inheretance from nn.Module from Pytorch Library
# The Code strucutre has been slightly altered to five better results

import os

import torch

# importing libraries to help with NN definations
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import *
from data import testloader, trainloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 1)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # FC Linear Layers
        self.fc1 = nn.Linear(12800, 64)
        self.fc2 = nn.Linear(64, 12)
        self.fc3 = nn.Linear(12, 10)

    def forward(
        self, x
    ):  # FORWARD FUNCTION FOR PROPOGATION OF INPUT DATA THROUGH THE NETWORK
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.pool(F.selu(self.conv3(x)))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = self.pool(F.selu(self.conv6(x)))

        x = x.reshape(
            x.shape[0], -1
        )  # print(x.shape)                          # TEMPORARY PRINT STATEMNETS I USED WHILE ALTERING THE NN STRUCTURE TO VERIFY THE IMAGE DIMENSTIONS AND MAINTAIN COMPATABILITY

        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_layer_embeds(
        self, num, x
    ):  # Series of Layer embeddings to help with visulization of feature maps genrated after covultion operation.
        if num == 1:
            return self.conv1(x)
        elif num == 2:
            return self.pool(self.conv1(x))
        elif num == 3:
            return self.conv2(x)
        elif num == 4:
            return self.pool(self.conv2(x))
        elif num == 5:
            return self.conv3(x)
        elif num == 6:
            return self.pool(self.conv3(x))
        elif num == 7:
            return self.conv4(x)
        elif num == 8:
            return self.pool(self.conv4(x))
        elif num == 9:
            return self.conv5(x)
        elif num == 10:
            return self.pool(self.conv5(x))
        elif num == 11:
            return self.conv6(x)
        elif num == 12:
            return self.pool(self.conv6(x))


if __name__ == "__main__":
    model = Net()
    model.to(device)  # Sending the Model to device for training

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Define a learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                print(
                    f"[{epoch + 1}, {i + 1}/{len(trainloader)}] loss: {running_loss / LOG_INTERVAL:.5f}"
                )
                running_loss = 0.0

        if ((epoch + 1) % VALIDATION_INTERVAL) == 0:
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"+++++" * 10)
            print("Finished Validation")
            print(f"Validation accuracy: {accuracy}%")
            print(f"+++++" * 10)

            # Learning rate scheduler step
            scheduler.step(running_loss)
    print(f"+++++" * 10)
    print("Finished Training")
    print(f"+++++" * 10)
    if os.path.isdir("./models"):
        pass
    else:
        os.mkdir("./models")
    torch.save(model.state_dict(), EXPORT_PATH)
