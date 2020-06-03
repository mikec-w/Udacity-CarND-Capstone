## Basic Model for Traffic Light Detection and Classification
## And training

import csv
from glob import glob
import cv2
import os
import numpy as np 
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import losses, optimizers, regularizers

## Simple CNN to get framework going...
# Load all the images
#img_path = '/home/student/Udacity/tl_classifier/training_data/light_classification/SIM/*/*.jpg'
img_path = '/home/student/Udacity/tl_classifier/training_data/light_classification/REAL/*/*.jpg'

#model_name = 'model_sim.h5'
model_name = 'model_real.h5'

# Create a list of all images
images = glob(img_path)

train_samples, validation_samples = train_test_split(images, test_size=0.2)

# Output has 4 states (red, yellow, green, unknown)
num_classes = 4

# Use a generator to load the images as required and label from filename
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []          
            states = []
            for batch_sample in batch_samples:
                path = batch_sample
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);
                resized = cv2.resize(image, (32,64))
                images.append(resized)
                # Get TL State
                if 'unknown' in path.lower():
                    states.append(3)
                elif 'green' in path.lower():
                    states.append(2)
                elif 'yellow' in path.lower():
                    states.append(1)
                elif 'red' in path.lower():
                    states.append(0)
                
            X_train = np.array(images)
            x_label = np.array(states)
            x_label = to_categorical(x_label, num_classes=num_classes)
            yield shuffle(X_train, x_label)

# Generator functions
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Basic Model
model = Sequential()

# PRE PROCESSING
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 32, 3)))

# Conv 1
# Reasonably high level of dropout due to overfitting becoming an issue.
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))

# Conv 2
model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))

# Conv 3
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
Dropout(0.65)

# Fully connected layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))

model.add(Dense(256, activation='relu'))
Dropout(0.65)
# Output to the 4 classes
model.add(Dense(num_classes, activation='softmax'))

# Define loss and optimizer
loss = losses.categorical_crossentropy
optimizer = optimizers.Adam()
          
model.compile(loss=loss, optimizer=optimizer)

# Fit
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
                    validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), \
                    epochs=20, verbose=1)

model.save(model_name)