#----------------------------------
# importing the libraries
#----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
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

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None) # path to trained model
parser.add_argument('--test_data_path', default=None) # path to test dataset

if __name__ == '__main__' : 

  args = parser.parse_args()
  args.model_dir = os.path.expanduser(args.model_path)
  args.test_data_dir = os.path.expanduser(args.test_data_path)
  
  model = Net()
  model = torch.load(args.model_dir) #'road_type_classifier.pt')
  
  #-------------------------------
  # loading training dataset
  data = pd.read_csv(args.test_data_dir) #'../data/training.csv')
  data.head()
  
  # loading images
  test_img = []
  for img_name in tqdm(data['file_name']):
    image_path = img_name
    img = imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # convert RGBA to RGB
    img = img/255   # TODO
    test_img.append(img)
  
  test_x = np.array(test_img)
  test_y = data['label'].values
  
  final_test_data = []
  final_label_test = []
  for i in tqdm(range(test_x.shape[0])):
    final_test_data.append(test_x[i])
    final_label_test.append(test_y[i])
  
  final_test = np.array(final_test_data)
  final_label_test = np.array(final_label_test)
  
  # converting test images into torch format
  final_test = final_test.reshape(len(test_x), 3, 256, 256)    # number of test_file, channels, width, height
  final_test = torch.from_numpy(final_test)
  final_test = final_test.float()
  
  #----------------------------------------------------
  # check out model performance
  # make prediction for training and validation data
  torch.manual_seed(0)
  batch_size = 1
  
  # prediction for training set
  prediction = []
  target = []
  permutation = torch.randperm(final_test.size()[0])

  for i in tqdm(range(0, final_test.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = final_test[indices], final_label_test[indices]

    print('debug : image index is ', indices)
         
    with torch.no_grad():
        output = model(batch_x.cuda())

    softmax = torch.exp(output).cpu() #Returns a new tensor with the exponential of the elements of the input tensor input.
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    print('[debug] prediction : ', predictions)
    print('[debug] target : ', batch_y)
    prediction.append(predictions)
    target.append(batch_y)

  # training accuracy
  accuracy = []
  for i in range(len(prediction)):
    #ORG accuracy.append(accuracy_score(target[i].cpu(), prediction[i])) # for multi-class classification
    #breakpoint()
    if target[i] == prediction[i][0] : 
      accuracy_score = 1
    else : 
      accuracy_score = 0   
    accuracy.append(accuracy_score)
    
  print('test accuracy: \t', np.average(accuracy))
  



  