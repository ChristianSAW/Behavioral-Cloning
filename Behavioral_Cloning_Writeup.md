# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: notebook_images/break_vs_time_full.png "Visualization of data set break vs. time"
[image2]: notebook_images/random_1by3_imgs.png "Visualizatin of random data frame"
[image3]: notebook_images/random_3by3_imgs.png "Visualization of random data frame"
[image4]: notebook_images/random_5x5_brightness_augment.png "Brightness augmentation"
[image5]: notebook_images/random_5x5_translation_augment.png "Translation augment"
[image6]: notebook_images/random_LCR_steering_corrected.png "Left Center and Right Images Included"
[image7]: notebook_images/random_preprocess.png "Preprocessing of random image with brightness factor of 0.35"
[image8]: notebook_images/random_preprocess_40.png "Preprocessing of random image with brightness factor of 0.40"
[image9]: notebook_images/random_training_process.png "Sample training generator image output"
[image10]: notebook_images/random_validation_process.png "Sample validation generator image output"
[image11]: notebook_images/speed_vs_time_full.png "Visualization of data set speed vs. time"
[image12]: notebook_images/speed_vs_time_partial.png "Visualization of data set speed vs. time (partial)"
[image13]: notebook_images/steering_vs_time_full.png "Visualization of data set steering vs. time"
[image14]: notebook_images/steering_vs_time_partial.png "Visualization of data set steering vs. time (partial)"
[image15]: notebook_images/throttle_vs_time_full.png "Visualization of data set throttle vs. time"
[image16]: notebook_images/throttle_vs_time_partial.png "Visualization of data set throttle vs. time (partial)"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model1u_PY_8.h5 containing a trained convolution neural network 
* Behavioral_Cloning_Writeup.md or summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model1u_PY_8.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with the following convolutional layers:
* Layer 0: 3 1x1 convolution
* Layer 1: 32 5x5 convolution
* Layer 2: 64 3x3 convolution
* Layer 3: 128 3x3 convolution

The following fully connected layers:
* Layer 4: fully connected with 512 outputs
* Layer 5: fully connected with 128 outputs 
* Layer 6: fully connected with 16 outputs 
* Layer 7: fully connected with 1 output

The model includes RELU layers to introduce nonlinearity in the convolutional layers and ELU activation functions after the fully connected layers. At the start of the model, the data is normalizedusing a Keras lambda layer. There is also max pooling after every convolutional layer.  

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The final model archetecture has a dropout layer after the activation of the 4th layer (fully connected) with a keep probability of 50%.  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I started with an initial learning rate of 0.0001. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the provided data and through data processing created data that would simulate recovery data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with what I believed to be a good archetecture and then iterate it. From the prevous project (and my ever building intuition), I started with the aformentioned archetecture. I knew I wanted to do much of the same data processing as before, so I anticipated a model of the complexity which I used in the traffic sign classifier project. Throutout testing, I tuned parameters at first to get a lower validation loss, but I quickly found I had to test every model on the simulator to truely test the effectiveness. Suprisingly, lower validation loss did not always correspond with a better track performance. 

My initial archetecture was essentially my final archetecture without as many dropout layers. Still, I made many mistakes allong the way of this project, and spent a long time getting the final model to work properly. I am still working on improving the model as I am not quite satisfied yet, but what I have finished satisfies this projects criteria. 

I additionally looked at the following articles initially to help determine my strategy: 
1) "https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9" by Vivek Yadav
2) "https://medium.com/deep-learning-turkey/behavioral-cloning-udacity-self-driving-car-project-generator-bottleneck-problem-in-using-gpu-182ee407dbc5" by Bahdir YILMAZ. 

I mainly got some ideas from Vivek and did my best to give hime credit in my code where I used his ideas. 

My strategy can be broken down as such: 
1. Determine data processing techniques to use
2. Determine model 
3. Develop training approach and respective generators (to speed up training)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. From the getgo, I used dropout layers, so I never ran into issues of overfitting (for the most part). 

I will talk about the data processing I did in section 3. 

For my training, I used an adam optimizer (as this worked relatively well in the past). During my hyper parameter tuning, I expiramented with using exponential decay as well as L2 Regulaization, but I found both to be unecessary and actually yield a worse test drive. 

I used generators to produce validation and training images and steering outputs to save on computation. 

**Validation Generator** <br>
This generator simpy applied the preprocessing (image resizing and cropping). I used batching rather than simply testing all of the validation data at once because I found using batching made training faster. Validation frames only used the center image as the neural network trained for center images. For each EPOCH, the validation data is shuffled.

Below is a random output for the validation generator (note that this is the same frame over and over again) 

![alt text][image10]

**Training Generator** <br>
This generator applied the data processing (brighness augmentation and translation) and preprocessing to images and steering outputs. For each batch, the data is shuffled and n (corresponding the batch size) random images and steering values are pulled from the training set and processed.

While the neural network was trained on center images for each frame, I used left and right images from frames to increase the training data set. I did this by shifting the steering data by a factor to turn the left or right image to a center image where the car was slightly shifted to the left or right and thus had a artificial steering value added to it to make it seem as if the car is trying to drive back into the center. I got this specific idea from Vivik Yadav, and used his correction factor. It worked quite well, and I did not tune this parameter. The impact of this was twofold: 1) I increaed my effective data set and 2) I added recovery data. 

Additionally, because the training data was from track 1, there was a bias of left turning. To combat this I ranomly flipped half of the training images (about the central y axis) as well as their steering values to counter the bias. 

Another bias that existed was the prevelance of low steering values (i.e. straight driving). To combat this, for each image selected, if the steering value was less tuned this value and found 50% to work quite well. 

I also expiramented with the training generator in selecting 2 images per frame or 1 image per frame. I found selecting 2 images yielded lower validation losses as well as better test track runs. This is probably because by selecting 2, I guaranteed that for every frame a recovery image (image which is not in the center and steering is focused on getting back to center) is used. 

Below is an image showing sample outputs from the training generator (note that is 1 frame that has been processed several times to show how different each processing can be). 

![alt text][image9]

**Parameter Tuning** <br>
Once I selected my initial model I tuned the following parameters: 
1. Convolutional layer sizes and filter number
2. Fully connected layer output numbers
3. Probability of keeping images from frames with steering values less than 0.15
4. inclusion of dropout and after which layers. 
5. EPOCH 

My initial guesses for convolutional layers sizes and filter number proved to work quite well. That being said, I know there a numerous combinations which would work, especially as my model had several layers. This was a simmilar situation with fully connected neuron output number. I did find however, during the test runs, that I could not lower my output neurons for my fully connected layers without soiling my test drives. 

I started with a arbitrary value of 75% for the probability of keeping images with steering values less than 0.15. I tried values lower and higher and setteled with 0.5 (50%) because it resulted in the lowest training validation losses and test runs. 

I broadly tested where to incldue dropout layers. Suprisingly, I found that the car would not complete a full lap of the first track if I had too many dropout layers. This was expecially true when adding dropout layers after the fully connected layers. From testing, I found including one after the first fully connected layer to midigate overfitting as well as allow the car to complete a run around the track. Also, considering the large amount of data I had, I knew overfitting would not be a large problem. 

Unlike with the traffic sign classifier, much fewer EPOCHs used. I started with 10, and found out that for my model, 8 EPOCHs was sufficient. With many variations of my model, as fews as 2 EPOCHs was sufficient. I did not tune the batch size, and arbitrarily used 256. In the future, this is another parameter I could tune. 

When I initially trained the model, I could not get the car to complete a single lap aound the whole track. A mistake I made with my generator resulting in each batch to be a size of 1 image set me back for quite a long time. I found that my model struggled the most with road which did not have strongly defined boarders especially in track 1 after the bridge when the track has a split with a dirt road. 

After much tuninig, I was able to get a model wher the car would complete the entire track. 

#### 2. Final Model Architecture

The final model archetecture is as follows:
* Lambda normalization function 
* Layer 0: 3 1x1 convolutions
* relu activation function
* 2x2 maxpooling 
* Layer 1: 32 5x5 convolutions
* relu activation function
* 2x2 maxpooling
* Layer 2: 64 3x3 convolutions 
* relu activation function
* 2x2 maxpooling 
* Layer 3: 128 3x3 convolutions 
* relu activation function
* 2x2 maxpooling
* Flatten Layer 
* Layer 4: fully connected with 512 outputs 
* elu activation function
* dropout with keep probability of 50%
* Layer 5: fully connected with 128 outputs 
* elu activation function
* Layer 6: fully connected with 16 outputs 
* elu activation function
* Layer 7: fully connected with 1 output

#### 3. Creation of the Training Set & Training Process

I initially tried to capture my own data. Unfortunately, I could not gather good data as I struggled to control the simulator well. So, I stuck with the provided data and seeing that others have been able to sucessfully train a model using just that, I knew I could too. The given training data was focused on keeping the car at the center of the track. 

Before I completed any data processing, I did some analysis of my data. 

There were 24108 images making up 8036 frames. With my 20% split. This meant there were 19286 images for training and 4822 images for validation. 

I made several historgrams of various outputs vs. time (speed, brake, throttle, steering). Because steering was the only value I was training for in this model, it became the most important, but I will plot the others here for you to see as well. 

**Steering vs. Time**<br>
![alt text][image13]

And a look at the first 250 frames to get a better idea of what these values are:

![alt text][image14]

**Speed vs. Time**<br>
![alt text][image11]

And a look at the first 250 frames to get a better idea of what these values are:

![alt text][image12]

**Throttle vs. Time**<br>
![alt text][image15]

And a look at the first 250 frames to get a better idea of what these values are:

![alt text][image16]


**Break vs. Time**<br>
![alt text][image1]

From this data, it is clear that there is a lot of data with low steering values confirming the bias towards straight driving. It is also interesting to note that the throttle is mostly being used with the excpetion of several times during the middle time stamps. Breaking is also almost entirely unised. Furthermore, this data was collected at high speed (30 mph). 

An example frame can be seen below,

![alt text][image2]

#### Data Processing: 

**Brightness Augmentation**<br>
I added brighness augmentations to simulate shadows on the road and eliminate the impact of shadows on the network. I did this by simply fudging the saturation value of each image. Below are sample images of this impact: 

![alt text][image4]

**Translations**<br> 
I added translations in the up/down and left/right to help simulate the car being on different positions in the road. This was also important to simulate recovery data as shifted image to the left or right make the car look like it is closer to the left or right. When shifting left or right I modified the steering value for the same reason I modified steering values when adding left and right images of frames. See the code for the exact implimentation of this. I give credit to Vivek Yadav for this augmentation as I learned to do it from him. Below are sample images of the translation impact. 

![alt text][image5]

#### Preprocessing:

To eliminate the effects of unimportant pixles in each image, I cropped out the hood of the car as well as much of the backgorund (sky, trees, etc.) by cutting out the top 30% of the image pixles as well as the bottom 25 pixles. 

I also reshaped the images to 63x63x3 to reduce the amount of pixles which needed to be trained to reduce training time. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
