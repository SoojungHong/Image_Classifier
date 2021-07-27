#----------------------------------
# importing the libraries
#----------------------------------
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from torchsummary import summary
from skimage.io import imread, imsave
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import os
import cv2
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import argparse
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', default=None) # path to training.csv file 


if __name__ == '__main__':

  args = parser.parse_args()
  args.data_dir = os.path.expanduser(args.csv_path)
  training_csv = args.data_dir + 'training.csv'
    
  #-------------------------------
  # loading training dataset

  data = pd.read_csv(training_csv)  #'../data/training.csv')
  data.head()
  #print(data)
  #print('debug : length(data) ', len(data))
  number_of_file = len(data)
  
  #-------------------------------
  # loading images
  train_img = []
  for img_name in tqdm(data['file_name']):
    image_path = img_name
    img = imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # convert RGBA to RGB
    img = img/255
    train_img.append(img)

  train_x = np.array(train_img)
  train_y = data['label'].values

  #---------------------------------
  # split training and validation
  train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)
    
  #-----------------------
  # training data
  final_train_data = []
  final_label_train = []
  for i in tqdm(range(train_x.shape[0])):
    final_train_data.append(train_x[i])
    final_label_train.append(train_y[i])

  final_train = np.array(final_train_data)
  final_label_train = np.array(final_label_train)
  num_train = len(final_train)
  # converting training images into torch format
  final_train = final_train.reshape(num_train, 3, 256, 256)    # number of file, channels, width, height
  final_train = torch.from_numpy(final_train) #creates a Tensor from a numpy.ndarray
  final_train = final_train.float()

  # converting the target into torch format
  final_label_train = final_label_train.astype(int)
  final_label_train = torch.from_numpy(final_label_train) # from ndarray to Tensor
  final_label_train = final_label_train.long()
  
  #---------------------
  # model definition
  model = Net()

  # defining the optimizer
  optimizer = Adam(model.parameters(), lr=0.000075)
  # defining the loss function
  criterion = CrossEntropyLoss()
  # checking if GPU is available
  if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

  print(model)
  
  
  #------------------------
  # training model

  torch.manual_seed(0)

  # batch size of the model
  batch_size = 16

  # number of epochs to train the model
  n_epochs = 100

  for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    permutation = torch.randperm(final_train.size()[0])
    training_loss = []

    for i in tqdm(range(0, final_train.size()[0], batch_size)):

        indices = permutation[i:i + batch_size]
        batch_x, batch_y = final_train[indices], final_label_train[indices]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
  

  #------------------
  # save model
  torch.save(model, 'road_type_classifier.pt')

  





    
  

  