#==========================================
# prepare training data csv file
# (example) - label should start with 0 
# intersection 0 
# junction 1
# straight 2
#==========================================

import os
import csv
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', default=None) # path to training.csv file 
parser.add_argument('--intersection_data', default=None) # path to (augmented) intersection type images
parser.add_argument('--none_intersection_data', default=None) # path to (augmented) none intersection type images


def traverse_images_and_add(directory, label, csv_writer):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(os.path.join(directory, filename))
            file_value_pair = []

            image_file = os.path.join(directory, filename)
            file_value_pair.append(image_file)
            file_value_pair.append(label)
            csv_writer.writerow(file_value_pair)
        else:
            continue


def read_training_csv(file_name):
    data = pd.read_csv(file_name)
    data.head()
    print(data)



if __name__ == '__main__':

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.csv_path)
    training_csv = args.data_dir + 'training.csv'
    
    # make csv file for training
    f = open(training_csv, 'w', encoding='utf-8')
    writer = csv.writer(f)
    head = ['file_name', 'label']
    writer.writerow(head)

    inter_dir = args.intersection_data #'..\data\intersection_augmented'
    #junction_dir = '..\data\junction'
    #straight_dir = '..\data\straight'
    none_inter_dir = args.none_intersection_data

    INTERSECTION_LABEL = 0
    #JUNCTION_LABEL = 2
    #STRAIGHT_LABEL = 3
    NONE_LABEL = 1

    # read intersection data and add to training.csv
    traverse_images_and_add(inter_dir, INTERSECTION_LABEL, writer)

    # read junction data and add to training.csv
    #traverse_images_and_add(junction_dir, JUNCTION_LABEL, writer)

    # read straight data and add to training.csv
    #traverse_images_and_add(straight_dir, STRAIGHT_LABEL, writer)
    
    # read none intersection data and add to training.csv 
    traverse_images_and_add(none_inter_dir, NONE_LABEL, writer)

    # close the file
    f.close()

    read_training_csv(training_csv) #'..\data\\training.csv')


