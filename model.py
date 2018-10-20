## IMPORT USEFUL PACKAGES
import os 
import pandas as pd
import numpy as np 
import cv2
import math
import sklearn
import csv

from sklearn.model_selection import train_test_split
from random import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import ELU
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
import time

### LOADING IN AND READING DATA:
dataPath = '../udacity-track1-data/driving_log.csv'

### Read CSV File
# Read Data from CSV File
data_full = pd.read_csv(dataPath, 
                        index_col = False)
data_full['direction'] = pd.Series('s', index=data_full.index)
print('Sucessfully accessed csv file')


# Define a image function loading the input image. 
# *Note, will not normalize the image input until put image through pipeline. No point in doing it until then. 
# Output: Image in RGB
def getImg(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

# Edit the path for reading the images in the right directory
def getPath(local_path):
    filename = local_path.split("/")[-1]
    host_path = '../udacity-track1-data/IMG/'+filename
    # print(host_path)
    return host_path

### Data Processing Scripts:
def brightness_augment(img):
    # convert to HSV
    img_aug = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_aug = np.array(img_aug, dtype = np.float64)    # convert to float64 for proper HSV conversion

    
    # random brightness
    random_bright = 0.35+np.random.uniform()
    
    # apply brightness augmentation
    img_aug[:,:,2] = img_aug[:,:,2]*random_bright
    img_aug[:,:,2][img_aug[:,:,2] > 254] = 255         # cap the maximum brightness to 255
    #img_aug[:,:,2][img_aug[:,:,2] < 30] = 30           # limit the darkest pixels
    #img_aug[:,:,2] = 50                                # testing

    # convert image back to RGB image
    img_aug = np.array(img_aug, dtype = np.uint8)     # needed for proper conversion back to RGB or else it wont work
    img_aug = cv2.cvtColor(img_aug, cv2.COLOR_HSV2RGB)
    
    return img_aug

def translate_img(img,steer,trans_range):
    trans_range_y = 10
    
    # Get Translations 
    trans_x = trans_range*np.random.uniform() - trans_range/2             # x - Left/Right
    trans_y = trans_range_y*np.random.uniform() - trans_range_y/2         # y - Up/Down
    
    # Update Steering Value
    steer_trans = steer + trans_x/trans_range*2*0.2
    
    # Create Translation Matrix
    T_M = np.float32([[1,0,trans_x],[0,1,trans_y]])
    
    # Apply Translation
    img_trans = cv2.warpAffine(img,T_M,(img.shape[1],img.shape[0]))       # translate image. Need (col, row) parameter. 
    
    return steer_trans, img_trans

def preprocess_img(img):
    img_shape = img.shape            # Shape output: [row,col,channel]
    
    top_crop = 0.35                   # % of top of image to crop out
    bottom_crop = 25                 # number of bottom pixles to crop out
    # Crop out unwanted sections of image: 
    pp_img = img[math.floor(img_shape[0]*top_crop):(img_shape[0]-bottom_crop),0:img_shape[1]]
    
    new_col = 64
    new_row = 64
    # Resize Image 
    pp_img = cv2.resize(pp_img,(new_col,new_row), interpolation=cv2.INTER_AREA)
    
    return pp_img

# Input is a row from the data_full csv file. 
# Returns 2 of 3 images and steering values 
# For training data
# *Note: Vivek Yadav added an additional correction factor to y_steer (multiplied all steering outputs by 1.2). 
# During testing, see if adding such a correction will improve accuracy. 
def process_train_img(data_row, use_brightness, use_translation, trans_range): 
        
    correction_factor = 0.25
    
    # Random combination of left, right, and center images
    rand_combo = np.random.randint(3)
    
    if (rand_combo == 0):                                      # Left and Center
        path_1 = getPath(data_row['left'][0].strip())
        path_2 = getPath(data_row['center'][0].strip())
        steer_cf_1 = correction_factor
        steer_cf_2 = 0.0
        
    if (rand_combo == 1):                                      # Right and Center
        path_1 = getPath(data_row['right'][0].strip())
        path_2 = getPath(data_row['center'][0].strip())
        steer_cf_1 = -correction_factor
        steer_cf_2 = 0.0
        
    if (rand_combo == 2):                                      # Left and Right
        path_1 = getPath(data_row['left'][0].strip())
        path_2 = getPath(data_row['right'][0].strip())
        steer_cf_1 = correction_factor
        steer_cf_2 = -correction_factor
        
    # Get Images 
    x_img_1 = getImg(path_1)
    x_img_2 = getImg(path_2)
    
    # Update Steering 
    y_steer_1 = data_row.steering[0] + steer_cf_1
    y_steer_2 = data_row.steering[0] + steer_cf_1
    
    # Brightness 
    if use_brightness:
        x_img_1 = brightness_augment(x_img_1)
        x_img_2 = brightness_augment(x_img_2)
        
    # Translation 
    if use_translation:
        y_steer_1, x_img_1 = translate_img(x_img_1, y_steer_1, trans_range)
        y_steer_2, x_img_2 = translate_img(x_img_2, y_steer_2, trans_range)

    # Preprocess
    x_img_1 = np.array(preprocess_img(x_img_1))
    x_img_2 = np.array(preprocess_img(x_img_2))
    
    # Flip 
    flip_1, flip_2 = np.random.randint(2), np.random.randint(2)
    
    if flip_1 == 0:
        x_img_1 = cv2.flip(x_img_1,1)
        y_steer_1 = -y_steer_1
        
    if flip_2 == 0:
        x_img_2 = cv2.flip(x_img_2,1)
        y_steer_2 = -y_steer_2
    
    return x_img_1, x_img_2, y_steer_1, y_steer_2

# Used for validation or testing data of data_full format. In my case, I only use this for training.
# There is a seperate preprocessing script within drive.py for preprocessing images from the simulator before they 
# are put in the neural net!
# Input Images: Center images from Frame. 
def process_predict_img(data_row):
    
    # Get Image and Steer
    x_img = getImg(getPath(data_row['center'][0].strip()))
    y_steer = data_row.steering[0]
    
    # Preprocess
    x_img = np.array(preprocess_img(x_img))
    
    return x_img, y_steer


## SPLIT DATA INTO TRAINING AND VALIDATION DATA SETS 
# Create Index List of Training and Validation Frames
input_list = np.linspace(0,data_full.shape[0]-1,data_full.shape[0],dtype = int)

# Split Data
train_list, valid_list = train_test_split(input_list,test_size = 0.2)

                
## KERAS GENERATOR AND SUBSAMPLING
# Note that the number of images within the batch will be double of the batch size input because 2 images are pulled
# for each frame
def generate_train_batch(data, train_list, pr_keep, use_brightness, use_translation, trans_range, batch_size = 32):
    new_row = 64
    new_col = 64
    thresh = 0.15
    
    batch_size_n = int(np.round(2*batch_size))
    
    # Create placeholder outputs (np Arrays)
    batch_img = np.zeros((batch_size_n, new_row, new_col,3))
    batch_steering = np.zeros(batch_size_n)
    
    # Start infinate loop
    while 1:
        # Shuffle list each time batch is called
        shuffle(train_list)
        for i_batch in range(batch_size):
            cont = True
            # Continue Loop Until Pick Values Which Work:
            while cont:
                
                # Get Random data_row from training list
                i = np.random.randint(len(train_list))                # Pull Index from List
                index_train = train_list[i]                           # Get data_row with pulled index 
                data_row = data.iloc[[index_train]].reset_index()
            
                # Process Images 
                x1, x2, y1, y2 = process_train_img(data_row, use_brightness, use_translation, trans_range)
                
                # Generate random num and check if steering values are above threshold. 
                randVal = np.random.uniform()
                if ((abs(float(y1)) < thresh) or (abs(float(y1)) < thresh)):
                    # if randVal is above probability thresh, throw away selection
                    if randVal > pr_keep:                                      
                        cont = True 
                    else:
                        cont = False
                else:
                    cont = False 
            
            # Add images and steering values to batch
            batch_img[(2*i_batch)] = x1
            batch_img[(2*i_batch)+1] = x2
            batch_steering[(2*i_batch)] = y1
            batch_steering[(2*i_batch)+1] = y2
            
            yield batch_img, batch_steering
            
# Note that the number of images within the batch will be double of the batch size input because 2 images are pulled
# for each frame
def generate_train_1img_batch(data, train_list, pr_keep, use_brightness, use_translation, trans_range, batch_size = 32):
    new_row = 64
    new_col = 64
    thresh = 0.15
    
    batch_size_n = int(np.round(2*batch_size))
    # Create placeholder outputs (np Arrays)
    batch_img = np.zeros((batch_size_n, new_row, new_col,3))
    batch_steering = np.zeros(batch_size_n)
    
    # Start infinate loop
    while 1:
        # Shuffle list each time batch is called
        shuffle(train_list)
        for i_batch in range(batch_size_n):
            cont = True
            # Continue Loop Until Pick Values Which Work:
            while cont:
                # Get Random data_row from training list
                i = np.random.randint(len(train_list))                # Pull Index from List
                index_train = train_list[i]                           # Get data_row with pulled index 
                data_row = data.iloc[[index_train]].reset_index()
   
                # Process Images 
                x1, x2, y1, y2 = process_train_img(data_row, use_brightness, use_translation, trans_range)
                
                # Generate random num and check if steering values are above threshold. 
                randVal = np.random.uniform()
                if (abs(float(y1)) < thresh):
                    # if randVal is above probability thresh, throw away selection
                    if randVal > pr_keep:                                      
                        cont = True 
                    else:
                        cont = False
                else:
                    cont = False 
            # Add images and steering values to batch
            batch_img[i_batch] = x1
            batch_steering[i_batch] = y1
            
            yield batch_img, batch_steering

def generate_valid(data, valid_list):
    new_row = 64
    new_col = 64
    
    # Create placeholder outputs (np Arrays)
    valid_img = np.zeros((len(data), new_row, new_col,3))
    valid_steering = np.zeros(len(data))
    
    # Start infinate loop
    while 1:
        # Shuffle list each time batch is called
        shuffle(valid_list)
        
        # Iterate through each valid center image in the data set. 
        for i in range(len(valid_list)):
            
            index_valid = valid_list[i]                              # Pull Index from List
            data_row = data.iloc[[index_valid]].reset_index()        # Get data_row with pulled index 
            x, y = process_predict_img(data_row)
            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            valid_img[i] = x
            valid_steering[i] = y
            
        yield valid_img, valid_steering

def generate_valid_batch(data, batch_size, valid_list):
    new_row = 64
    new_col = 64
    
    # Create placeholder outputs (np Arrays)
    valid_img = np.zeros((batch_size, new_row, new_col,3))
    valid_steering = np.zeros(batch_size)
    
    # Shuffle list before each training run to not influence anything. 
    shuffle(valid_list)
    
    # Start infinate loop
    while 1:
        # Iterate through each valid center image in the data set. 
        for i in range(batch_size):
            index_valid = valid_list[i]                              # Pull Index from List
            data_row = data.iloc[[index_valid]].reset_index()        # Get data_row with pulled index 
            x, y = process_predict_img(data_row)
            
            #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            #y = np.array([[y]])
            valid_img[i] = x
            valid_steering[i] = y
            
        yield valid_img, valid_steering
        
## TRAINING 
def model1():  
    
    ## GENERAL PARAMETERS:
    new_row = 64
    new_col = 64
    channels = 3
    input_shape = (new_row, new_col, channels)
    pooling_size = (2,2)
    dropout_keep_prob = 0.5
    
    ## SPECIFIC LAYER PARAMETERS: 
    # Layer 0: 1x1 Convolutional
    filter_size_0 = 1
    filter_num_0 = 3
    name_conv0= 'conv0' 
    # Layer 1: Convolutional 
    filter_size_1 = 5
    filter_num_1 = 32
    name_conv1 = 'conv1'
    # Layer 2: Convolutional
    filter_size_2 = 5
    filter_num_2 = 64
    name_conv2 = 'conv2' 
    # Layer 3: Convolutional
    filter_size_3 = 3
    filter_num_3 = 128
    name_conv3 = 'conv3' 
    # Layer 4: Fully Connected 
    fc_out_1 = 512
    name_fc1 = 'fc1' 
    # Layer 5: Fully Connected 
    fc_out_2 = 128
    name_fc2 = 'fc2'
    # Layer 6: Fully Connected
    fc_out_3 = 16
    name_fc3 = 'fc3'
    
    # Layer 7: Fully Connected (LAST LAYER MUST HAVE 1 OUTPUT)
    fc_out_4 = 1
    name_fc4 = 'fc4' 
    
    ## PIPELINE 
    model = Sequential()
    
    # Normalization
    model.add(Lambda((lambda x: x/255.0 - 0.5), input_shape=input_shape, output_shape=input_shape))
    
    # Layer 0: 1x1x3 Convolutional
    model.add(Conv2D(filter_num_0,
                     (filter_size_0, filter_size_0),
                     name = name_conv0))
    model.add(Activation('relu'))                              # Activation: RELU
    model.add(MaxPooling2D(pool_size=pooling_size))            # Pooling
    model.add(Dropout(dropout_keep_prob))                      # Dropout
    
    # Layer 1: Convolutional
    model.add(Conv2D(filter_num_1,
                     (filter_size_1, filter_size_1),
                     name = name_conv1))
    model.add(Activation('relu'))                              # Activation: RELU
    model.add(MaxPooling2D(pool_size=pooling_size))            # Pooling
    model.add(Dropout(dropout_keep_prob))                      # Dropout
    
    # Layer 2: Convolutional
    model.add(Conv2D(filter_num_2,
                     (filter_size_2, filter_size_2),
                     name = name_conv2))
    model.add(Activation('relu'))                              # Activation: RELU
    model.add(MaxPooling2D(pool_size=pooling_size))            # Pooling
    model.add(Dropout(dropout_keep_prob))                      # Dropout
    
    # Layer 3: Convolutional
    model.add(Conv2D(filter_num_3,
                     (filter_size_3, filter_size_3),
                     name = name_conv3))
    model.add(Activation('relu'))                              # Activation: RELU
    model.add(MaxPooling2D(pool_size=pooling_size))            # Pooling
    model.add(Dropout(dropout_keep_prob))                      # Dropout
    # Flatten
    model.add(Flatten())
    
    # Layer 4: Fully Connected
    model.add(Dense(fc_out_1,
                   name=name_fc1,
                   kernel_initializer='he_normal'))
    model.add(ELU())                                           # Activation: ELU
    model.add(Dropout(dropout_keep_prob))                      # Dropout
    
    # Layer 5: Fully Connected
    model.add(Dense(fc_out_2,
                   name=name_fc2,
                   kernel_initializer='he_normal'))
    model.add(ELU())                                           # Activation: ELU
    model.add(Dropout(dropout_keep_prob))                      # Dropout
    
    # Layer 6: Fully Connected
    model.add(Dense(fc_out_3,
                   name=name_fc3,
                   kernel_initializer='he_normal'))
    model.add(ELU())                                           # Activation: ELU
    model.add(Dropout(dropout_keep_prob))                      # Dropout

    # Layer 7: Fully Connected (OUTPUT SIZE = 1)
    model.add(Dense(fc_out_4,
                   name=name_fc4,
                   kernel_initializer='he_normal'))
    return model

# Use a Model you know that works to test the rest of your training pipeline and see if there is a problem somewhere
def model2a():
    ## GENERAL PARAMETERS:
    new_row = 64
    new_col = 64
    channels = 3
    input_shape = (new_row, new_col, channels)
    pooling_size = (2,2)
    dropout_keep_prob = 0.5
    
    ## PIPELINE 
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=input_shape,output_shape=input_shape))
    model.add(Conv2D(36, (5, 5), name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(48, (3, 3), name='conv5', padding='valid'))
    model.add(Flatten())
    model.add(Dense(100, name="hidden1", kernel_initializer="he_normal"))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(50,name='hidden2', kernel_initializer="he_normal"))
    model.add(ELU())
    model.add(Dense(10,name='hidden3',kernel_initializer="he_normal"))
    model.add(ELU())
    model.add(Dense(1, name='output', kernel_initializer="he_normal"))
    return model

## HYPERPARAMETERS 
batch_size = 256
EPOCH = 8
EPOCH_inner = 10                            # EPOCH between parameter/hyper parameter changes. 
beta = 0.00005                              # L2 Regularization scaling factor
learning_rate = 0.0001

## PARAMETERS
best_EPOCH = 0                              # Keeping Track of Best Epoch
best_valid_loss = 100                       # Arbitrarily High Validation Loss 
use_brightness = True
use_translation = True 
trans_range = 80
pr_keep = 0.75

## Training Data:
train_size = len(train_list)
## Validation Data:
valid_size = len(valid_list)

## Optimizer
model = model2a()                            # CHANGE THIS FOR TRAINING DIFFERENT MODELS
adam = Adam(lr=learning_rate)
model.compile(optimizer=adam,
             loss='mse')

## Generator
changeParam = False
useValidBatch = True 
useSingleImg_Train = False

if useValidBatch:
    valid_generator = generate_valid_batch(data_full, batch_size, valid_list)
else:
    valid_generator = generate_valid(data_full, valid_list)

if not changeParam:
    if useSingleImg_Train:
        train_generator = generate_train_1img_batch(data_full, train_list, pr_keep, use_brightness, use_translation,
                                                    trans_range, batch_size)
    else:
        train_generator = generate_train_batch(data_full, train_list, pr_keep, use_brightness, use_translation,
                                               trans_range, batch_size)

## BEGIN TRAINING 
model1 = False
model2 = True

# Loop Through Each EPOCH and train. 
# Each EPOCH is basically retraining with initialization of weights and parameters from the previous EPOCH
if model1:
    # Loop Through Each EPOCH and train. 
    # Each EPOCH is basically retraining with initialization of weights and parameters from the previous EPOCH
    version = 'b'
    print("Training...")
    print()

    start_time = time.time()           # time training
    for i_EPOCH in range(EPOCH):
        if not changeParam:
            if useSingleImg_Train:
                train_generator = generate_train_1img_batch(data_full, train_list, pr_keep, use_brightness, use_translation,
                                                            trans_range, batch_size)
            else:
                train_generator = generate_train_batch(data_full, train_list, pr_keep, use_brightness, use_translation,
                                                       trans_range, batch_size)
    
        num_steps_train = np.round(len(train_list)/(batch_size))
        num_steps_valid = np.round(len(valid_list)/(batch_size)) # Should be able to just have as 1 bc each batch is entire validation set
    
        # Record history when training to plot later 
        history_object = model.fit_generator(train_generator,
                                             steps_per_epoch = num_steps_train,
                                             epochs = EPOCH_inner,
                                             verbose = 1,
                                             callbacks = None,
                                             validation_data = valid_generator,
                                             validation_steps = num_steps_valid)
        # Check for best EPOCH
        print('Got To Validation Loss Check')
        valid_loss = history_object.history['val_loss'][0]
    
        if (valid_loss < best_valid_loss):
        
            # Update Best EPOCH/Loss
            best_EPOCH = i_EPOCH
            best_valid_loss = valid_loss
        
            # Save Best EPOCH
            saveName = 'model1_' + str(version) + '_EPOCH_' + str(i_EPOCH+1) + '.h5'
            model.save(saveName)
            
        # Update Parameters 
        if changeParam:
            # Update keep probability for low steering value files
            pr_keep = 1/(i_EPOCH+1)
        
    end_time = time.time()
    time_diff = end_time - start_time
    print("Total training Time: ", np.round(time_diff,2), "s")

if model2:
    # Loop Through Each EPOCH and train. 
    # Each EPOCH is basically retraining with initialization of weights and parameters from the previous EPOCH
    modelNum = '2'
    version = 'c'
    print("Training...")
    print()
    start_time = time.time()           # time training
    
    ### INSERT CODE HERE
    saveName = 'model' + str(modelNum) + str(version) + '_{epoch:03d}.h5'
    saveName2 = 'model' + str(modelNum) + str(version) + '_' + str(EPOCH_inner) + '.h5'
    checkpoint = ModelCheckpoint(saveName,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    
    num_steps_train = np.round(len(train_list)/(batch_size))
    #num_steps_train = len(train_list)

    if useValidBatch:
        num_steps_valid = np.round(len(valid_list)/(batch_size)) 
    else:
        num_steps_valid = 1
    #num_steps_valid = len(valid_list)

    # Record history when training to plot later 
    history_object = model.fit_generator(train_generator,
                                         steps_per_epoch = num_steps_train,
                                         epochs = EPOCH_inner,
                                         verbose = 1,
                                         callbacks = [checkpoint],
                                         validation_data = valid_generator,
                                         validation_steps = num_steps_valid)
    model.save(saveName2)
    ### END OF TRAINING CODE
    end_time = time.time()
    time_diff = end_time - start_time
    print("Total training Time: ", np.round(time_diff,2), "s")