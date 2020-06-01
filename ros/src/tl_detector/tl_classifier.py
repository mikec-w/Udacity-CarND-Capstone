## Code for classifying a traffic light from the image...

import os
import numpy as np
import tensorflow as tf
import time
import cv2
import h5py

import datetime

class TLClassifier(object)

    model = None

    def __init__(self):

        # Load Model
        f = h5py.File("model.h5", mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

        model = load_model(args.model)

    def get_classification(self, image, wp = 0):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        TL_State = model.predict(image)
        
        return TL_State
