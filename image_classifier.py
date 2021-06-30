#reference : https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212


#---------------------------------------------
# Import libraries
#---------------------------------------------
import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural networks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer

from src.models import *


#----------------------------------------------
# Load and normalize data
#----------------------------------------------
# python image library of range [0, 1]
# transform them to tensors of normalized range[-1, 1]

transform = transforms.Compose( # composing several transforms together
            [transforms.ToTensor(), # to tensor object
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

# set batch_size
batch_size = 4

# set number of workers
num_workers = 4

# load train data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # ORG is download=True
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

"""
# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
"""

# put 10 classes into a set
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#------------------------
# Visualize images
#------------------------
def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


if __name__ == '__main__':
    # get random training images with iter function
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # call function on our images
    imshow(torchvision.utils.make_grid(images))

    # print the class of the image
    print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

    net = Net()
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #------------------------------
    # train the network
    #------------------------------
    #ORG start = torch.cuda.Event(enable_timing=True)
    #ORG end = torch.cuda.Event(enable_timing=True)

    #ORG start.record()

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # whatever you are timing goes here
    #ORG end.record()

    # Waits for everything to finish running
    #ORG torch.cuda.synchronize()

    print('Finished Training')
    #ORG print(start.elapsed_time(end))  # milliseconds

    # savePATH = './cifar_net.pth'
    PATH = './SAVED_MODEL/cifar_net.pth'
    torch.save(net.state_dict(), PATH)  # reloadnet = Net()
    net.load_state_dict(torch.load(PATH))
