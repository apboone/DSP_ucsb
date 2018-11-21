from __future__ import print_function, division
import torch
import torchvision
import torchvision.transforms as transforms

import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

data_path_train = '/Users/alexanderboone/Desktop/test/testdata/'
data_path_test = '/Users/alexanderboone/Desktop/test/standards/300PxHopper/'


groundTruth = pd.read_csv('/Users/alexanderboone/Desktop/test/standards/StandardsFile.csv')

trainset = torchvision.datasets.ImageFolder(root=data_path_train, transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root=data_path_test,transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=3, shuffle=False, num_workers=2)

classes = ('Learned', 'Reverse Learned', 'Shortcut')

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


import torch.nn as nn
import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels = 6, kernel_size=5, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=300, out_channels=600, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.fc1 = nn.Linear(600, 600)
#         print("hit1")
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         print(x.shape)
#         x = x.view(-1, 600)
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = 300, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=300, out_channels = 300, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(300*3*3,300)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


print ('check point 3')

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
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


#class SimpleNet(nn.Module):
#     def __init__(self, num_classes=3):
#         super(SimpleNet, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()

#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()

#         self.pool = nn.MaxPool2d(kernel_size=2)

#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU()

#         self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
#         self.relu4 = nn.ReLU()

#         self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.relu1(output)

#         output = self.conv2(output)
#         output = self.relu2(output)

#         output = self.pool(output)

#         output = self.conv3(output)
#         output = self.relu3(output)

#         output = self.conv4(output)
#         output = self.relu4(output)

#         output = output.view(-1, 16 * 16 * 24)

#         output = self.fc(output)

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# outputs = net(images)



# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))



# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the xxxx test images: %d %%' % (
#     100 * correct / total))


