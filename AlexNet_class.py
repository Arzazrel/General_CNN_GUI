# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: file containing the class that reproduces the AlexNet model

description: network description at the end of the file
"""
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Model

# class that implement the AlexNet model
class AlexNet:
    # constructor
    """
    def __init__(self,class_number):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = 224                   # height of the images in input to CNN
        self.img_width = 224                    # width of the images in input to CNN
        self.img_channel = 3                    # channel of the images in input to CNN (RGB)
        """
        
    # constructor with image size of the input layer
    def __init__(self,class_number,img_width = 224,img_height = 224,img_channel = 3):
        self.model = None                       # var that will contain the model of the CNN AlexNet
        self.num_classes = class_number         # var that will contain the number of the classes of the problem (in our case is 2 (fire, no_fire))
        # var for the image dimension
        self.img_height = img_height            # height of the images in input to CNN
        self.img_width = img_width              # width of the images in input to CNN
        self.img_channel = img_channel          # channel of the images in input to CNN (RGB)
        
    # method for make the model of the CNN
    def make_model(self):
        self.model = models.Sequential()
        """
        inp = layers.Input(shape=(self.img_width, self.img_height, self.img_channel))       # input
        # 1st Conv layer (has Max pooling)
        conv_1 = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), padding='valid', activation='relu')(inp)
        conv_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(conv_1)       # Max pooling
        lrn_arg = {'depth_radius': 5, 'alpha': 10**-4, 'beta':0.75}
        print(conv_1.shape)
        conv_1 = layers.Lambda(tf.nn.local_response_normalization,arguments=lrn_arg)(conv_1)     # LRN Normalisation
        # 2nd Conv layer (has Max pooling)
        conv_2 = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='valid', activation='relu')(conv_1)
        conv_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid')(conv_2)       # Max pooling
        #conv_2 = tf.nn.local_response_normalization(conv_2,bias=2,depth_radius=5, alpha=10**-4, beta=0.75)     # LRN Normalisation
        # 3rd Conv layer (hasn't Max pooling)
        conv_3 = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu')(conv_2)
        #conv_3 = tf.nn.local_response_normalization(conv_3,bias=2,depth_radius=5, alpha=10**-4, beta=0.75)     # LRN Normalisation
        # 4th Conv layer (hasn't Max pooling)
        conv_4 = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu')(conv_3)
        #conv_4 = tf.nn.local_response_normalization(conv_4,bias=2,depth_radius=5, alpha=10**-4, beta=0.75)     # LRN Normalisation
        # 5th Conv layer (has Max pooling)
        conv_5 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu')(conv_4)
        conv_5 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid')(conv_5)       # Max pooling
        conv_5 = tf.nn.local_response_normalization(conv_5,bias=2,depth_radius=5, alpha=10**-4, beta=0.75)     # LRN Normalisation
        # 1th dense layer
        flat = layers.Flatten()(conv_5)
        net_1 = layers.Dense(4096, activation='relu')(flat)                                       # dense layer
        net_1 = layers.Dropout(0.5)(net_1)                                                        # dropout
        net_1 = tf.nn.local_response_normalization(net_1,bias=2,depth_radius=5, alpha=10**-4, beta=0.75)     # LRN Normalisation
        # 2nd dense layer
        net_2 = layers.Dense(4096, activation='relu')(net_1)                                       # dense layer
        net_2 = layers.Dropout(0.5)(net_2)                                                         # dropout
        net_2 = tf.nn.local_response_normalization(net_2,bias=2,depth_radius=5, alpha=10**-4, beta=0.75)     # LRN Normalisation
        # Output layer
        out = layers.Dense(self.num_classes, activation='softmax')(conv_1)
        
        self.model = Model(inputs = inp, outputs = out)             # assign the CNN in model
        
        """
        # version of the model made using batch normalisation
        # 1st Conv layer (has Max pooling)
        self.model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4,4), padding='valid', activation='relu', input_shape=(self.img_width, self.img_height, self.img_channel)))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisation
        # 2nd Conv layer (has Max pooling)
        self.model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))       # Max pooling
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 3rd Conv layer (hasn't Max pooling)
        self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 4th Conv layer (hasn't Max pooling)
        self.model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 5th Conv layer (has Max pooling)
        self.model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid'))       # Max pooling
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 1th dense layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(4096, activation='relu'))                                       # dense layer
        self.model.add(layers.Dropout(0.5))                                                         # dropout
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # 2nd dense layer
        self.model.add(layers.Dense(4096, activation='relu'))                                       # dense layer
        self.model.add(layers.Dropout(0.5))                                                         # dropout
        self.model.add(layers.BatchNormalization())                                                 # Batch Normalisations
        # Output layer
        self.model.add(layers.Dense(self.num_classes, activation='softmax'))                  
        
        self.model.summary()                                                                        # summary of the net
        
    # method for compile the model
    def compile_model(self):
        # compile rmsprop
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
    # method for return the model
    def return_model(self):
        return self.model
    
#AlexNet_Model = AlexNet(2)    # create an instance of the IfriNet class
#AlexNet_Model.make_model()  
        
"""
brief description:
    AlexNet is a CNN model designed by Alex Krizhevsky and Ilya Sutskever, under the supervision of Geoffrey Hinton. 
    AlexNet represented a fundamental breakthrough in image classification problems and won the ImageNet Large Scale Visual Recognition Challenge in 2012. 
    AlexNet has eight levels: the first five levels are convolutional, of which the first two use max-pooling, while the last three levels do not, being fully connected. The last three layers are simple fully connected layers.
    The network uses local Response Normalization (LRN) and the ReLU activation function, except for the last layer which uses Softmax activation function.
    With the exception of the last layer, the rest of the network was split into two copies, running separately on two GPUs. 
"""