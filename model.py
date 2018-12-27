#import libraries
import os
import csv
import cv2
import time
import random
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd


from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPool2D

#Variables delaration
Epochs = 5
BatchSize = 32
dataSource = "./CarND-Behavioral-Cloning-P3/data/"

#convert the image to RGB
def preprocess_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Function to load the image path and steering angle.
def get_image_data(dataSource):
    df = pd.read_csv(dataSource+"driving_log.csv")
    images = []
    angles = []
    for i in range(len(df)):
        image_c = dataSource+df["center"][i]
        sourcel = df["left"][i]
        sourcel = sourcel.strip()
        image_l = dataSource+sourcel
        sourcer = df["right"][i]
        sourcer = sourcer.strip()
        image_r = dataSource+sourcer

        
        angle_c = float(df["steering"][i])
        offset = 0.2
        angle_l = angle_c + offset
        angle_r = angle_c - offset

        # add image paths and angles to data set
        images.extend((image_c, image_l, image_r))
        angles.extend((angle_c, angle_l, angle_r))

    return images,angles

#Generator function code
def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)



imagePaths, measurements = get_image_data(dataSource)
#zipping the data to send it to the generator
samples = list(zip(imagePaths, measurements))
#Splitting the data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#No of saamples
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=BatchSize)
validation_generator = generator(validation_samples, batch_size=BatchSize)

#Model
model = Sequential()
model.add(Cropping2D(((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.) - 0.5))
model.add(Conv2D(24, (5,5), strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), padding="valid", activation="relu"))
model.add(Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu"))
model.add(Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= np.ceil(len(train_samples)/BatchSize), validation_data=validation_generator, nb_val_samples=np.ceil(len(validation_samples)/BatchSize), nb_epoch=5, verbose=1)

model.save("model.h5")

print(history_object.history.keys())
print(history_object.history['loss'])
print(history_object.history['val_loss'])

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'valiation set'], loc='upper right')
plt.savefig("Loss.png")
plt.show()