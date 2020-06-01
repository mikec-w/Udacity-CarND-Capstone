## Code for classifying a traffic light from the image...

import os
import numpy as np
import tensorflow as tf
import time
import cv2
import h5py
from keras.models import load_model
from keras import __version__ as keras_version

import datetime

class TLClassifier(object):

    model = None

    def __init__(self):

        # Load Model
        #f = h5py.File("../../../tl_classifier/model.h5", mode='r')

        self.model = load_model("../../../tl_classifier/model.h5")

    def get_classification(self, image, wp = 0):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        TL_State = self.model.predict(image)

        return TL_State
