from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import copy
import torch.nn as nn
import numpy as np

# TODO: Implement a convolutional neural network (https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)


class Net(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """

    def __init__(self):
        super(Net, self).__init__()
       # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(3, 32, 5)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(64*6*6, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = nn.functional.relu(x)
        # Run max pooling over x
        x = self.pool(x)

        # Pass data through conv2
        x = self.conv2(x)
        # Use the rectified-linear activation function over x
        x = nn.functional.relu(x)
        # Run max pooling over x
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        # Pass data through fc1
        x = self.fc1(x)
        x = nn.functional.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)

        return x


# TODO: You can change these data augmentation and normalization strategies for
#  better training and testing (https://pytorch.org/vision/stable/transforms.html)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dataset initialization
data_dir = 'data'  # Suppose the dataset is stored under this folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}  # Read train and test sets, respectively.

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

class_names = image_datasets['train'].classes


# Set device to "cpu" if you have no gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: Implement training and testing procedures (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)


def train_test(model, criterion, optimizer, scheduler, num_epochs=25):

    for epoch in range(num_epochs):
        print("\t\tEpoch: ", (epoch+1))
        running_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    return None


model_ft = Net()  # Model initialization

model_ft = model_ft.to(device)  # Move model to cpu

criterion = nn.CrossEntropyLoss()  # Loss function initialization

# TODO: Adjust the following hyper-parameters: learning rate, decay strategy, number of training epochs.
# Optimizer initialization
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft, step_size=20, gamma=0.5)  # 0.1  # Learning rate decay strategy


print("Running Device is: ", device)  # Print whether it is on CPU or GPU.
train_test(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
           num_epochs=15)


# This section of code prints the test labels and their predictions.
# This section has been commented out.
'''correct = 0
total = 0

with torch.no_grad():
    for i, (img, labels) in enumerate(dataloaders['test']):
        img, labels = img.to(device), labels.to(device)
        outputs = model_ft(img)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        print(f"Iteration {i},\tPrediction: {preds},\tActual: {labels}")
    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))'''


# This section of code prints the test labels and their probability accuracy.
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in dataloaders['test']:
        img, labels = data
        img, labels = img.to(device), labels.to(device)
        outputs = model_ft(img)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
over_accuracy = 0
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    over_accuracy += accuracy
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                         accuracy))
print(f"Overall Accuracy is : {over_accuracy/len(classes)}%")
