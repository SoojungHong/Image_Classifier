
# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import os
import cv2
import argparse


#----------------
# arguments 
#----------------
parser = argparse.ArgumentParser()
parser.add_argument('--original_data', default=None) # path to the original images 
parser.add_argument('--augmented_data', default=None) # path to the augmented images


#--------------------------------
# creating images by rotation
#--------------------------------
def rotate_image(file_name, path, out_path):
    image = io.imread(path+file_name)
    save_dir = out_path
    # displaying the image
    #io.imshow(image)
    #io.show()

    # rotate as many as possible
    for angle in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        rotated = rotate(image, angle=angle, mode='wrap')
        #io.imshow(rotated)
        #io.show()
        io.imsave(save_dir + str(angle) + '_rotated_' + file_name, rotated)


def shift_image(file_name, path, out_path):
    image = io.imread(path + file_name)
    save_dir = out_path

    # displaying the image
    # io.imshow(image)
    # io.show()
    rgbImage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # convert to RGB from RGBA

    # apply shift operation
    for shift_amount in [15, 25, 35, 45, 55, 65, 75, 85] :
        transform = AffineTransform(translation=(shift_amount, shift_amount))
        wrapShift = warp(rgbImage, transform, mode='wrap')

        #plt.imshow(wrapShift)
        #plt.show()
        plt.imsave(save_dir + str(shift_amount) + 'shifted_' + file_name, wrapShift)


def flip_image(file_name, path, out_path):
    image = io.imread(path + file_name)
    save_dir = out_path

    rgbImage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # convert to RGB from RGBA

    # flip image left-to-right
    flipLR = np.fliplr(rgbImage)
    plt.imsave(save_dir + 'flipped_left_right_' + file_name, flipLR)

    flipUD = np.flipud(rgbImage)
    plt.imsave(save_dir + 'flip_upside_down_' + file_name, flipUD)



if __name__ == '__main__':

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.original_data)
    args.out_data_dir = os.path.expanduser(args.augmented_data)
    in_directory = args.data_dir
    out_directory = args.out_data_dir

    # given directory, augment all files
    for filename in os.listdir(in_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(os.path.join(in_directory, filename))
            rotate_image(filename, in_directory, out_directory)
            shift_image(filename, in_directory, out_directory)
            flip_image(filename, in_directory, out_directory)
