# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: file containing the class that reproduces the GoogLeNet model

description: network description at the end of the file
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Model

# ------------------------------------ start: utility methods ------------------------------------

# utility function to implement the inception module in which 1×1, 3×3, 5×5 convolution and 3×3 max pooling are executed in parallel and their output is merged.
# in_net: is the input , fil_1x1: is the number of filters of conv 1x1 layer, the same for other similar fil
# fil_1x1_3x3: is the number of filters of the 1x1 reduction convolutionary layer before conv 3x3 and so on for others similar fil
# fil_m_pool: is the number of filter of the 1x1 convolutionary layer after max pooling 
def inception_mod(in_net, fil_1x1, fil_1x1_3x3, fil_3x3, fil_1x1_5x5, fil_5x5, fil_m_pool):
    # four parallel path
    
    path1 = layers.Conv2D(filters=fil_1x1, kernel_size=(1, 1), padding='same', activation='relu')(in_net)       # conv 1x1
    
    path2 = layers.Conv2D(filters=fil_1x1_3x3, kernel_size=(1, 1), padding='same', activation='relu')(in_net)   # conv 1x1 to reduce
    path2 = layers.Conv2D(filters=fil_3x3, kernel_size=(1, 1), padding='same', activation='relu')(path2)        # conv 3x3
    
    path3 = layers.Conv2D(filters=fil_1x1_5x5, kernel_size=(1, 1), padding='same', activation='relu')(in_net)   # conv 1x1 to reduce
    path3 = layers.Conv2D(filters=fil_5x5, kernel_size=(1, 1), padding='same', activation='relu')(path3)        # conv 5x5
    
    path4 = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(in_net)                          # max pool
    path4 = layers.Conv2D(filters=fil_m_pool, kernel_size=(1, 1), padding='same', activation='relu')(path4)     # conv 1x1 to reduce
    
    return tf.concat([path1, path2, path3, path4], axis=3)                  # merge of the different path

# ------------------------------------ end: utility methods ------------------------------------

# class that implement the GoogLeNet model
class GoogLeNet:
    
    # constructor
    def __init__(self,class_number):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = 224                   # height of the images in input to CNN
        self.img_width = 224                    # width of the images in input to CNN
        self.img_channel = 3                    # channel of the images in input to CNN (RGB)
        
        
    # method for make the model of the CNN. Due to the structure of the network (inception and parallel paths) it is better to use keras Models class instead of sequential
    def make_model(self):
        
        inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
        
        # seq_0: is the first part of the CNN network starting with the input and ending with the first auxiliary classifier
        seq_0 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inp)
        seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
        seq_0 = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(seq_0)
        seq_0 = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(seq_0)
        seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
        seq_0 = inception_mod(seq_0, fil_1x1=64, fil_1x1_3x3=96, fil_3x3=128, fil_1x1_5x5=16, fil_5x5=32, fil_m_pool=32)
        seq_0 = inception_mod(seq_0, fil_1x1=128, fil_1x1_3x3=128, fil_3x3=192, fil_1x1_5x5=32, fil_5x5=96, fil_m_pool=64)
        seq_0 = layers.MaxPooling2D(3, strides=2)(seq_0)
        seq_0 = inception_mod(seq_0, fil_1x1=192, fil_1x1_3x3=96, fil_3x3=208, fil_1x1_5x5=16, fil_5x5=48, fil_m_pool=64)
        
        # first auxiliary classifier
        aux_1 = layers.AveragePooling2D((5, 5), strides=3)(seq_0)
        aux_1 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux_1)
        aux_1 = layers.Flatten()(aux_1)
        aux_1 = layers.Dense(1024, activation='relu')(aux_1)
        aux_1 = layers.Dropout(0.7)(aux_1)
        aux_1 = layers.Dense(self.num_classes, activation='softmax',name = "aux_1")(aux_1)         # aux output layer
        
        # seq_1: is the second part of the CNN network starting with the end of seq_0 and ending with the second auxiliary classifier
        seq_1 = inception_mod(seq_0, fil_1x1=160, fil_1x1_3x3=112, fil_3x3=224, fil_1x1_5x5=24, fil_5x5=64, fil_m_pool=64)
        seq_1 = inception_mod(seq_1, fil_1x1=128, fil_1x1_3x3=128, fil_3x3=256, fil_1x1_5x5=24, fil_5x5=64, fil_m_pool=64)
        seq_1 = inception_mod(seq_1, fil_1x1=112, fil_1x1_3x3=144, fil_3x3=288, fil_1x1_5x5=32, fil_5x5=64, fil_m_pool=64)
        
        # second auxiliary classifier
        aux_2 = layers.AveragePooling2D((5, 5), strides=3)(seq_1)
        aux_2 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux_2)
        aux_2 = layers.Flatten()(aux_2)
        aux_2 = layers.Dense(1024, activation='relu')(aux_2)
        aux_2 = layers.Dropout(0.7)(aux_2) 
        aux_2 = layers.Dense(self.num_classes, activation='softmax',name = "aux_2")(aux_2)         # aux output layer
        
        # seq_2: is the last part of the CNN network starting with the end of seq_1 and ending with the end of CNN
        seq_2 = inception_mod(seq_1, fil_1x1=256, fil_1x1_3x3=160, fil_3x3=320, fil_1x1_5x5=32, fil_5x5=128, fil_m_pool=128)
        seq_2 = layers.MaxPooling2D(3, strides=2)(seq_2)
        seq_2 = inception_mod(seq_2, fil_1x1=256, fil_1x1_3x3=160, fil_3x3=320, fil_1x1_5x5=32, fil_5x5=128, fil_m_pool=128)
        seq_2 = inception_mod(seq_2, fil_1x1=384, fil_1x1_3x3=192, fil_3x3=384, fil_1x1_5x5=48, fil_5x5=128, fil_m_pool=128)
        seq_2 = layers.GlobalAveragePooling2D()(seq_2)
        seq_2 = layers.Dropout(0.4)(seq_2)
        out = layers.Dense(self.num_classes, activation='softmax', name = "out")(seq_2)           # output layer
        
        self.model = Model(inputs = inp, outputs = [out, aux_1, aux_2])             # assign the CNN in model
        self.model.summary()
                
    # method for compile the model
    def compile_model(self):
        self.model.compile(optimizer='adam', 
              loss=[losses.categorical_crossentropy,
                    losses.categorical_crossentropy,
                    losses.categorical_crossentropy],
              loss_weights=[1, 0.3, 0.3],
              metrics=['accuracy'])
        
    # method for return the model
    def return_model(self):
        return self.model

"""
brief description:
    GoogLeNet won ILSVRC-2014 and is one of the most successful models of the earlier years of CNN.
    GoogLeNet was created by Google Inc. and the model was published in the paper "Going Deeper with Convolutions".
    It uses different types of methods that allow it to create a deeper architecture. 
    
    1×1 convolution: These convolutions are used to decrease the number of parameters (weights and biases) of the architecture and increase the depth of the architecture. 
    Global Average Pooling: Fully connected layers contain most of the parameters of many architectures, which causes an increase in computing cost. 
                            In the GoogLeNet architecture, a method called global average pooling is used at the end of the network. This level takes a feature map of 7×7 and the average of 1×1. 
                            This reduces the number of trainable parameters to 0 and improves accuracy.
    Inception module: In this module the 1×1, 3×3, 5×5 convolution and 3×3 max pooling are executed in parallel to the input and the output of these is stacked together to generate the final output. 
                      The idea behind this is that convolution filters of different sizes handle multiple scale objects better.
    The overall architecture has a depth of 22 layers. The architecture also contains two auxiliary classifier layers connected to the output via inception layers.
    This architecture takes images of size 224 x 224 with RGB colour channels. All convolutions in this architecture use rectified linear units (ReLUs) as activation functions.

The architectural details of the auxiliary classifiers are as follows:
- A medium pooling layer with filter size 5×5 and step 3.
- A 1×1 convolution with 128 filters for size reduction and ReLU activation.
- A fully connected layer with 1025 outputs and ReLU activation.
- Dropout adjustment with dropout ratio = 0.7
- A softmax classifier with 1000 output classes similar to the main softmax classifier.
    
"""