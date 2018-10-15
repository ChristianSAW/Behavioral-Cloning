## IMPORT USEFUL PACKAGES
import numpy as np
import cv2
import math
import sklearn
import csv


### LOADING IN AND READING DATA:
dataPath = '/opt/carmd_p3/data'

# Edit the path for reading the images in the right directory
def path_editor(local_path):
    filename = local_path.split("/")[-1]
    host_path = 'data/IMG/'+filename
    # print(host_path)
    return host_path

# Read the image the split the image into training and validation dataset

samples = []
count=0
turn_angle_threshold = 0.15
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    camera = ['center', 'left', 'right']
    for line in reader:
        # Disregard the tags in .csv file
        if line[0] == 'center':
            continue
        # Drop 75% of images whose steering angle is less than 0.15
        token = np.random.uniform()
        if abs(float(line[3])) < turn_angle_threshold:
            if token > 0.75:
                for i in range(3):
                    count += 1
                    samples.append([camera[i], line[i], line[3]])
            else:
                continue
        else:
            for i in range(3):
                count += 1
                samples.append([camera[i], line[i], line[3]])
                
## SPLIT DATA INTO TRAINING AND VALIDATION DATA SETS 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
from random import shuffle


## Simple model okay

