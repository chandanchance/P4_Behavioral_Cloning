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

[image1]: ./writeupImages/trainingTheModel.PNG "Model Visualization"
[image2]: ./writeupImages/training.PNG "Training Screen Shot"
[image3]: ./writeupImages/lossGraph.PNG "Loss visualization"
[image4]: ./writeupImages/carview.PNG "Car view"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

model.py contains the modeling code.
model.h5 contains the output model from the code.
video.mp4 contains the video convertion of the drive.
drive.py contains the code to run the model in the simulator

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

### About Data 

80% of the data is used as training data and the remaining 20% is used to validate the model.
The shape of an input image is (160, 320, 3), i.e. 160 x 320 RGB pixel
The output is steering angle value

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with varing depths and filter sizes (model.py lines 96-109) 

The data is normalized in the model using a Keras lambda layer (code line 98). RELU function is used as an activation function.

The architechture is as follows:

Input Layer (160,320,3) with Cropping2D ((50,20),(0,0)) --> Normalization --> CNN 24 @ 5x5 -->CNN 36 @ 5x5 -->CNN 48 @ 5x5 -->CNN 64 @ 5x5 -->CNN 64 @ 5x5 --> Dropout 0.5 --> Flatten Layer --> Dense 100 --> Dense 50 --> Dense 10 --> Dense 1


This is almost similar to Nvidia CNN architechture but I have added dropout to avoid overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I used 5 Epochs on the first go to check with the losses.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).
No of epochs used = 5
validation size = 0.2
loss was calculated based on mse

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  'get_camera_offset' function was used to select the offset angle (This was initially tried). Based on the revious review suggestion I changed the offset to 0.2

case 1 : Flipping of the data was done randomly, this would also help in reducing the over fitting.
This caused few features to go uncaptured so I decided to flip all the images which doubled the size. (The code to flip randomly selected images is commented.)


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA

![alt text][image1]

The layers consists of :

Input Layer (160,320,3) with Cropping2D ((50,20),(0,0)) --> Normalization --> CNN 24 @ 5x5 -->CNN 36 @ 5x5 -->CNN 48 @ 5x5 -->CNN 64 @ 5x5 -->CNN 64 @ 5x5 --> Dropout 0.5 --> Flatten Layer --> Dense 100 --> Dense 50 --> Dense 10 --> Dense 1

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I used dropout layer and shuffling the data images.

The images were converted to YUV from BGR ( minimize variance in light and shadow by using histogram equalization). To try I converted the BGR to RGB and this performed better. The off set angle was added to the left and right camera images.
The images were flipped randomly first and then all the images were flipped.

These images were converted to numpy array and feed to the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.To improve the driving behavior in these cases, I Cropped the images from (75,25) to (50,20) and this was a game changer.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture


Input Layer (160,320,3) with Cropping2D ((50,20),(0,0)) --> Normalization --> CNN 24 @ 5x5 -->CNN 36 @ 5x5 -->CNN 48 @ 5x5 -->CNN 64 @ 5x5 -->CNN 64 @ 5x5 --> Dropout 0.5 --> Flatten Layer --> Dense 100 --> Dense 50 --> Dense 10 --> Dense 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the given data. I also tried modeling over the data which I uploaded but the model ended up in over fitting the images so I removed them.

![alt text][image4]

After the collection process and preprocessing ie., adding steering angles and flipping the images.I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the losses. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Training the model 

Training screen shot:

![alt text][image2]

![alt text][image3]

This is the comparision of the training and the validation loss.

Cropping the image to (50,20) helped to avoid unwanted learnings.


