#----------------------------------
# importing the libraries
#----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import skimage.io as io
import torch

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
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None) # path to trained model
parser.add_argument('--raw_data_path', default=None) # path to data that contains all types of road images
parser.add_argument('--train_data_path', default=None) # path to train data that is selected as particular road type


if __name__ == '__main__' : 
  args = parser.parse_args()
  args.model_path = os.path.expanduser(args.model_path)
  args.raw_data_path = os.path.expanduser(args.raw_data_path)
  args.train_data_path = os.path.expanduser(args.train_data_path)
  
  model = Net()
  model = torch.load(args.model_path)
  
  #road all images in raw train data folder
  for filename in os.listdir(args.raw_data_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
      # loading images
      img = imread(args.raw_data_path+filename) # return type is image
      img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # convert RGBA to RGB
      img = img/255
      image_x = img.reshape(1, 3, 256, 256) 
      
      # converting test images into torch format
      final_test = torch.from_numpy(image_x)
      final_test = final_test.float()
      
      with torch.no_grad():
        output = model(final_test.cuda()) # predict (classify)
        softmax = torch.exp(output).cpu() #Returns a new tensor with the exponential of the elements of the input tensor input.
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        print('[debug] prediction : ', predictions)

        if predictions[0] == 0 : # classified as multilanes (class 0)
          print('road type is multilanes')
          #save this file to selected dataset
          io.imsave(args.train_data_path + filename, img)

    