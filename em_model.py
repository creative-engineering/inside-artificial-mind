
from __future__ import division, absolute_import
import re
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
import random
import sys
import tflearn.helpers.summarizer as s
import tensorflow as tf

class EMR:

  def __init__(self):
  	self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

  def build_network(self):
      print("---> Starting Neural Network") 
      self.network = input_data(shape = [None, 48, 48, 1], name='input')
      
      layer1 = conv_2d(self.network, 64, 5, activation = 'relu')
      self.network = max_pool_2d(layer1, 3, strides = 2)
      
      self.network = conv_2d(self.network, 64, 5, activation = 'relu')
      self.network = max_pool_2d(self.network, 3, strides = 2)
      
      self.network = conv_2d(self.network, 128, 4, activation = 'relu')
      self.network = dropout(self.network, 0.3)
      
      layer3 = fully_connected(self.network, 3072, activation = 'relu')
      self.network = fully_connected(layer3, len(self.target_classes), activation = 'softmax')
      
      
      self.network = regression(self.network,
        optimizer = 'momentum',
        loss = 'categorical_crossentropy',
        name='targets' )
      
      self.model = tflearn.DNN(
        self.network,
        checkpoint_path = 'model_1_nimish',
        max_checkpoints = 1,
        tensorboard_verbose = 0
      )
      
      self.model2 = tflearn.DNN(
        layer3,
        checkpoint_path = 'model_1_nimish',
        max_checkpoints = 1,
        tensorboard_verbose = 0
      )
      
      self.model3 = tflearn.DNN(
        layer1,
        checkpoint_path = 'model_1_nimish',
        max_checkpoints = 1,
        tensorboard_verbose = 0
      )
      
      self.load_model()

  def predict(self, image):
    if image is 0:
      return None
    
    image = image.reshape([-1, 48, 48, 1])
    return self.model.predict(image)

  def predict2(self, image):
    if image is 0:
      return None
    
    image = image.reshape([-1, 48, 48, 1])
    return self.model2.predict(image)
    
  def predict3(self, image):
    if image is 0:
        return None
    
    image = image.reshape([-1, 48, 48, 1])
    return self.model3.predict(image)


  def load_model(self):
    if isfile("model_1_nimish.tflearn.meta"):
      self.model.load("model_1_nimish.tflearn")
      print('---> Loading moodel from:- model_1_nimish.tflearn')
    else:
        print("---> Couldn't find model model_1_nimish.tflearn")

if __name__ == "__main__":
  network = EMR()
  import run


