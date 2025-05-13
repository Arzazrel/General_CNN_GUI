# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:07:58 2024

@author: Alessandro Diana
"""

# general
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'    # to manage better the allocator of the VRAM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                # 0 = all logs, 1 = warnings, 2 = errors, 3 = nothing
import numpy as np
import random
import time
import math
import shutil
# for model
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
# for plot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# for image visualization
import cv2
import PIL
from PIL import ImageTk
from PIL import Image
# for GUI
from tkinter import *
from tkinter import ttk
# for thread
from threading import Thread
# for tensorflow-GPU
from tensorflow.python.client import device_lib 
# import of my files
from net_classes import AlexNet_class as ANet
from net_classes import GoogLeNet_class as GLNet
from net_classes import IfritNet_class as IfritNet

# ------------------------------------ start: global var ------------------------------------
# ---- GUI variables ----
window = Tk()
toggle_duplicate_ds = BooleanVar(value=False)   # var to indicate if create a copy of dataset with the indicated shape for the images(default False)
toggle_grey_scale = BooleanVar(value=False)     # var to indicate if convert or not in greyscale images a copy of dataset with the indicated shape for the images(default False)
window_width = 1800                             # is the width of the tkinter window
window_height = 750                             # is the height of the tkinter window
# explain frame
ex_f_padx = 10                                  # horizontal pad for the explain frame
ex_f_pady = 5                                   # vertical pad for the explain frame
ex_frame_width = window_width - 2*ex_f_padx     # width of the explain frame
ex_frame_height = 45                            # height of the explain frame
# top frame
t_f_padx = 10                                   # horizontal pad for the top frame
t_f_pady = 5                                    # vertical pad for the top frame
top_frame_width = window_width - 2*t_f_padx     # width of the top frame
top_frame_height = 180                          # height of the top frame
# image frame
im_f_padx = 10                                  # horizontal pad for the image frame
im_f_pady = 5                                   # vertical pad for the image frame
im_frame_width = window_width - 2*im_f_padx     # width of the image frame
im_frame_height = 250                           # height of the image frame
# bottom frame
b_f_padx = 10                                   # horizontal pad for the bottom frame
b_f_pady = 5                                    # vertical pad for the bottom frame
b_frame_width = window_width - 2*im_f_padx      # width of the bottom frame
b_frame_height = 170                            # height of the bottom frame
# error frame
er_f_padx = 10                                  # horizontal pad for the error frame
er_f_pady = 5                                   # vertical pad for the error frame
er_frame_width = window_width - 2*im_f_padx     # width of the error frame
er_frame_height = 45                            # height of the error frame

# ---- errors variables ----
error_text = StringVar()                        # text that shows the errors
error_text.set('')                              # default value: empty text
er_load_model_text = "Please insert a model name in the field before load CNN model."   # error text that occur when try to load a model without specifying the CNN name model
er_load_model_unknown_text = "There isn't a CNN model with the specified name."         # error text that occur when it's not possible return the specified model
er_save_model_text = "Please insert a model name in the field before save CNN model."   # error text that occur when try to save a model without specifying the CNN name model
er_no_ds_text = "Please load the dataset and retry."                                    # error text that occur when user want to work with dataset without loading one
er_no_ext_ds_text = "Please load the external test dataset and retry."                  # error text that occur when user want to work with external dataset without loading this
er_train_without_ds_text = "Before train the model you must load image dataset."        # error text that occur when user want to make and fit the CNN model without loading dataset
er_eval_without_model_text = "Before evaluate the model you must make and fit or load a mode."  # error text that occur when user want to evaluate the model without make and fit the CNN model
er_no_model_specified_text = "Please chose a CNN model or load one before fit CNN model."       # error text that occur when user want to make the model without chose one
er_predict_text = "Before predict you must train model and load an image."              # error text that occur when user want to predict without take image or train model
er_format_epoch_text = "Error format in the Number of epochs input, you must insert a positive number, please retry."   # error text that occur when user insert a incorrect number of epochs format
er_format_batch_size_text = "Error format in the Number of batch_size input, you must insert a positive number, please retry."   # error text that occur when user insert a incorrect number of batch_size format
er_format_image_size_text = "Error format in the Numbers for the image size, you must insert a positive number, please retry."   # error text that occur when user insert a incorrect number for image size
er_format_early_text = "Error format in the Number of early patience input, you must insert a positive number, please retry."   # error text that occur when user insert a incorrect number of early patience format
er_down_ds_text = "Dataset downloading already started, please wait."                   # error text that occur when user want to download the dataset when the program is downloading ds

# ---- status variables ----
model_trained = False                           # variable that show if there is a model trained
image_to_visualize = None                       # image that will be visualized in the GUI
index_image_visualized = -1                     # index of the image visualized in GUI, default value is -1
ds_downloading = False                          # indicate if the pogram is downloading the dataset
ext_ds_downloading = False                      # indicate if the pogram is downloading the extern dataset
status_DS_text = StringVar()                            # text that shows the state of the dataset (missing, loading, loaded)
status_DS_text.set('Image DataSet: missing')            # the default value is 'missing'
status_ext_test_DS_text = StringVar()                   # text that shows the state of the extern test dataset (missing, loading, loaded)
status_ext_test_DS_text.set('External test DS: missing')   # the default value is 'missing'
CNN_menu_text = StringVar()                             # text that shows in the menu the type of CNN model select, the possible values are (None, AlexNet, GoogleNet). The chosen model can be train and fit
CNN_menu_text.set('None')                               # the default value is 'None'
status_model_text = StringVar()                         # text that shows the state of the CNN model (empty, trained)
status_model_text.set('CNN model: empty')               # the default value is 'empty'

# ---- label text variables ----
classify_text = StringVar()                     # text that shows the class of the new object classified by classifier
classify_text.set('')                           # default value
classify_ext_text = StringVar()                 # text that shows the class of the new object classified by classifier when the image is taken from external test ds
classify_ext_text.set('')                       # default value
label_image_text = StringVar()                  # text that shows the groundtruth of the visualised image
label_image_text.set('')                        # default value
label_ext_image_text = StringVar()              # text that shows the groundtruth of the visualised image when it is taken from external test dataset
label_ext_image_text.set('')                    # default value

# ---- path variables ----
path_dir_ds = os.path.join("Dataset","polmonite")           # folder in which there are the image ds for training
path_dir_test_ds = os.path.join("Dataset","pkm_test_ds")    # folder in which there are the image ds for testing
path_dir_model = "Model"                                    # folder in which there are saved the CNN model
path_check_point_model = os.path.join(path_dir_model,"train_hdf5")  # folder in which there are saved the checkpoint for the model training
res_copy_ds_name = "res_ds_copy"                # the name of the folder to contain the resized copy of the dataset (case of timing optimization)
# ---- model variables ----
network = None                                  # contain the CNN model, default value is None
truncate_set = False                            # variable which indicates whether the sets (train, test,val) must be truncate or not when divided to batch_size
# indicate the param of the past model if all are the same of the actual make_model request the program doesn't make again the model
past_param_model = { "type_model": None, "epochs": None, "batch_size": None, "early_patience": None}                           
batch_size = 32                                 # batch size for training, this is the default value
def_img_height = 224                            # default value for height of the images in input to CNN 
def_img_width = 224                             # default value for width of the images in input to CNN 
def_img_channel = 3                             # default value for channel of the images in input to CNN 
img_height = def_img_height                     # height of the images in input to CNN 
img_width = def_img_width                       # width of the images in input to CNN 
img_channel = def_img_channel                   # channel of the images in input to CNN            
output_activation = 'softmax'                   # activation function of the output layer  
hidden_activation = 'relu'                      # activation function of the hidden layer
epochs = 100                                    # number of epochs for training, this is the deafault value
early_patience = 10                             # number of epochs with no improvement after which training will be stopped 

# ---- dataset variables ----
classes = []                        # the label associated with each class will be the position that the class name will have in this array
total_image_ds = []                 # contain the total image dataset
total_labels_ds = []                # contain the labels of the total image dataset
train_image = []                    # contain the images choosen as train set in fit
train_label = []                    # contain the labels of the images choosen as train set in fit
val_img = []                        # contain the images choosen as validation set in fit
val_label = []                      # contain the labels of the images choosen as validation set in fit
test_image = []                     # contain the images choosen as test set in evaluation
test_label = []                     # contain the labels of the images choosen as test set in evaluation
test_image_ext = []                 # contain the images choosen as test set in evaluation import from extern test set
test_label_ext = []                 # contain the labels of the images choosen as test set in evaluation import from extern test set
test_set_split = 0.2                # test set size as a percentage of the total dataset
val_set_split = 0.1                 # validation set size as a percentage of the training set

# -------------------------- start: dataset_information_variable --------------------------
# ---- status variable for dataset information ----
del_corrupt_img = False                         # variable that indicate to the program whether it should delete any corrupted images in the dataset
ds_analysing = False                            # indicate if the pogram is analysing the dataset for tak ethe information for the report
RGB_mean_dict = {}                              # contain for each class the value for the color Blue , Green, Red. For a quick comparison of the colour distribution in the various classes and dataset.
                                                # the dictionary will have a key for each class and the corresponding value will be an array of three values.
# -------------------------- end: dataset_information_variable --------------------------
# ------------------------------------ end: global var ------------------------------------

# ------------------------------------ start: methods for GUI ------------------------------------
# method that cleans GUI elements
def cleanGUI():
    list = window.grid_slaves()
    for l in list:
        l.destroy()

# method for handling the closing of the window by the user
def on_closing():
    # close window
    window.destroy()
    
# method executed for change the view visualised
def current_view_to_visualise():
    cleanGUI()                              # clean all element of the GUI
    # create variable for the GUI elements
    explainText = "Welcom to genCNNClassifier.\nA general program to train and test CNNs."
    CNN_model_text = ['None','AlexNet','GoogleNet','Ifrit_1','Ifrit_2','Ifrit_3','Ifrit_4']

    # create the GUI elements and place them 
    # ---- start: explain_frame ----
    explain_frame = Frame(window, width=ex_frame_width , height=ex_frame_height , bg='grey')
    explain_frame.grid(row=0, column=0, padx=ex_f_padx , pady=ex_f_pady , sticky="nsew")
    explain_frame.grid_propagate(False)
    
    explainTextLabel = Label(explain_frame, text=explainText)               # Label to briefly explain
    explainTextLabel.grid(row=0, column=0, sticky="W", padx=window_width/2, pady=5)
    # ---- end: explain_frame ----
    
    # ---- start: top frame (contain: import for dataset, import and save for model, select and fit model buttons) ----
    top_frame = Frame(window, width=top_frame_width , height=top_frame_height , bg='grey')
    top_frame.grid(row=1, column=0, padx=t_f_padx , pady=t_f_pady , sticky="nsew")
    top_frame.grid_propagate(False)
    
    # -- start: row 0 --
    # ---- for image size -- start ----
    path_ds_label = Label(top_frame, text="DS name (in 'dataset' folder): ")    # label for the name of the DS
    path_ds_label.grid(row=0, column=0, sticky="W", padx=5, pady=10) 
    
    path_ds_input = Entry(top_frame, width=15)                                  # entry for the width of the images in input to CNN 
    path_ds_input.grid(row=0, column=1, sticky="WE", padx=5)
    
    image_size_label = Label(top_frame, text="Input image size: ")              # label for image size of the CNN model (image of the input layer)
    image_size_label.grid(row=0, column=2, sticky="W", padx=5, pady=10) 
    
    image_width_input = Entry(top_frame, width=15)                              # entry for the width of the images in input to CNN 
    image_width_input.grid(row=0, column=3, sticky="WE", padx=5)
    
    image_height_input = Entry(top_frame, width=15)                             # entry for the height of the images in input to CNN 
    image_height_input.grid(row=0, column=4, sticky="WE", padx=5)
    
    image_channel_input = Entry(top_frame, width=15)                            # entry for the channels of the images in input to CNN
    image_channel_input.grid(row=0, column=5, sticky="WE", padx=5)
    
    def_image_size_label = Label(top_frame, text="Default size: (" + str(def_img_width) + "," + str(def_img_height) + "," + str(def_img_channel) + ")")              # label for image size of the CNN model (image of the input layer)
    def_image_size_label.grid(row=0, column=6, sticky="W", padx=5, pady=10) 
    # ---- for image size -- end ----
    
    button_analyse_ds = Button(top_frame, text="Analyse DS", command=lambda: btn_analyse_ds(path_ds_input.get(), image_width_input.get(),image_height_input.get(),image_channel_input.get()))   # button to load the whole dataset
    button_analyse_ds.grid(row=0, column=8, sticky="W", padx=5, pady=10)
    
    toggle = Checkbutton(top_frame, text="Resized DS copy", variable=toggle_duplicate_ds)       # checkbox for the copy of dataset
    toggle.grid(row=0, column=9, sticky="W", padx=5, pady=10)
    
    toggle_gs = Checkbutton(top_frame, text="Greyscale DS copy", variable=toggle_grey_scale)    # checkbox for the copy of dataset
    toggle_gs.grid(row=0, column=10, sticky="W", padx=5, pady=10)
    
    btn_load_ds = Button(top_frame, text="Load image DS", command=lambda: btn_load_ds_method(image_width_input.get(),image_height_input.get(),image_channel_input.get()))   # button to load the whole dataset
    btn_load_ds.grid(row=0, column=11, sticky="W", padx=5, pady=10)
     
    # -- end: row 0 --
    
    # -- start: row 1 --
    name_model_label = Label(top_frame, text="CNN model name: ")                # label for the name of the CNN model to load or to save
    name_model_label.grid(row=1, column=6, sticky="W", padx=5, pady=10) 
    
    name_model_input = Entry(top_frame, width=15)                               # entry for the CNN model name
    name_model_input.grid(row=1, column=7, sticky="WE", padx=5)
    
    btn_load_model = Button(top_frame, text="Load CNN model", command=lambda: load_saved_model(name_model_input.get()))   # button to load the CNN model
    btn_load_model.grid(row=1, column=8, sticky="W", padx=5, pady=10)
    
    btn_save_model = Button(top_frame, text="Save CNN model", command=lambda: save_model(name_model_input.get()))   # button to save the CNN model
    btn_save_model.grid(row=1, column=9, sticky="W", padx=5, pady=10)

    btn_fit_model = Button(top_frame, text="Fit CNN model", command=lambda: make_fit_model(CNN_menu_text.get(),number_epochs_input.get(),batch_size_input.get(),early_stopping_input.get(),num_train_input.get()))      # button to fit CNN model
    btn_fit_model.grid(row=1, column=12, sticky="W", padx=5, pady=10)
    # -- end: row 1 --
    
    # -- start: row 2 --
    name_model_label = Label(top_frame, text="Select CNN model:")           # label to explain the choice of CNN models
    name_model_label.grid(row=2, column=0, sticky="W", padx=5, pady=10) 
    
    CNN_menu = OptionMenu(top_frame, CNN_menu_text,*CNN_model_text)         # creating select menu for CNN model
    CNN_menu.grid(row=2, column=1,padx=10)
    
    epoch_label = Label(top_frame, text="Number of epoch:")                 # label to explain the number of epoch
    epoch_label.grid(row=2, column=2, sticky="W", padx=5, pady=10) 
    
    number_epochs_input = Entry(top_frame, width=10)                        # entry for the number of epochs
    number_epochs_input.grid(row=2, column=3, sticky="WE", padx=5)

    batch_size_label = Label(top_frame, text="Batch size:")                 # label to explain the batch size
    batch_size_label.grid(row=2, column=4, sticky="W", padx=5, pady=10) 
    
    batch_size_input = Entry(top_frame, width=10)                           # entry for the number batch size
    batch_size_input.grid(row=2, column=5, sticky="WE", padx=5)
    
    early_stopping_label = Label(top_frame, text="Early patience:")         # label to explain the early patience
    early_stopping_label.grid(row=2, column=6, sticky="W", padx=5, pady=10) 
    
    early_stopping_input = Entry(top_frame, width=10)                       # entry for the number of early patience
    early_stopping_input.grid(row=2, column=7, sticky="WE", padx=5)
    
    num_train_label = Label(top_frame, text="Number of train to do:")       # label to explain the number of train to do (statistic test)
    num_train_label.grid(row=2, column=8, sticky="W", padx=5, pady=10) 
    
    num_train_input = Entry(top_frame, width=10)                            # entry for the number of train to do (statistic test)
    num_train_input.grid(row=2, column=9, sticky="WE", padx=5)
    # -- end: row 2 --
    
    # -- start: row 3 -- status variable --
    model_label = Label(top_frame, textvariable=status_model_text)              # label for the model (missing,loading,loaded)
    model_label.grid(row=3, column=2, sticky="W", padx=5, pady=10)  

    dataset_label = Label(top_frame, textvariable=status_DS_text)               # label for the status of DS (missing,loading,loaded)
    dataset_label.grid(row=3, column=3, sticky="W", padx=5, pady=10)   
    # -- end: row 3 --
    # ---- end: top frame ----
    
    # ---- start: image frame ----
    image_frame = Frame(window, width=im_frame_width , height=im_frame_height , bg='grey')
    image_frame.grid(row=2, column=0, padx=im_f_padx , pady=im_f_pady , sticky="nsew")
    image_frame.grid_propagate(False)
    
    # image take by test set that will be predicted
    if image_to_visualize is not None:
        image_label = Label(image_frame, image= image_to_visualize)
        image_label.grid(row=0, column=1, sticky="W", padx=(window_width/2 - img_width/2), pady=5)
    # ---- end: image frame ----

    # ---- start: bottom frame (contains: button to perform load an image of the test set and predict) ----
    bottom_frame = Frame(window, width=b_frame_width , height=b_frame_height , bg='grey')
    bottom_frame.grid(row=3, column=0, padx=b_f_padx , pady=b_f_pady , sticky="nsew")
    bottom_frame.grid_propagate(False)
    
    # -- start: row 0 --
    btn_load_random_image = Button(bottom_frame, text="Take image", command=btn_load_image) # buttons to load a random image from test DS to predict 
    btn_load_random_image.grid(row=0, column=0, sticky="W", padx=10, pady=10)
    
    label_text = Label(bottom_frame, text="Label: ", width=9)                      # label for the groundtruth
    label_text.grid(row=0, column=1, sticky="W", padx=10, pady=10)
    
    correct_label = Label(bottom_frame, textvariable=label_image_text, width=9)                      # label for the groundtruth
    correct_label.grid(row=0, column=2, sticky="W", padx=10, pady=10)
    
    btn_predict = Button(bottom_frame, text="Classify:", command=predict, width=11)                    # button to predict label of image
    btn_predict.grid(row=0, column=3)
    
    result_classifier_label = Label(bottom_frame, textvariable=classify_text, width=9)               # label for the result of classifier
    result_classifier_label.grid(row=0, column=4, sticky="W", padx=10, pady=5)
    # -- end: row 0 --
    
    # -- start: row 1 --
    btn_load_test_ds = Button(bottom_frame, text="Load extern test DS", command=lambda: btn_load_ext_ds_method(image_width_input.get(),image_height_input.get(),image_channel_input.get()))   # button to load the whole extern test dataset
    btn_load_test_ds.grid(row=1, column=0, sticky="W", padx=10, pady=10)
    
    dataset_test_label = Label(bottom_frame, textvariable=status_ext_test_DS_text)           # label for the status of DS (missing,loading,loaded)
    dataset_test_label.grid(row=1, column=5, sticky="W", padx=10, pady=10)
    # -- end: row 1 --
    
    # -- start: row 2 --
    btn_load_random_test_img = Button(bottom_frame, text="Take extern test image", command=btn_load_ext_image) # buttons to load a random image from test DS to predict 
    btn_load_random_test_img.grid(row=2, column=0, sticky="W", padx=10, pady=10)
    
    ext_test_label = Label(bottom_frame, text="Label: ", width=9)                      # label for the groundtruth
    ext_test_label.grid(row=2, column=1, sticky="W", padx=10, pady=10)
    
    correct_ext_test_label = Label(bottom_frame, textvariable=label_ext_image_text, width=9)                      # label for the groundtruth
    correct_ext_test_label.grid(row=2, column=2, sticky="W", padx=10, pady=10)
    
    btn_predict = Button(bottom_frame, text="Classify:", command=predict, width=11)                    # button to predict label of image
    btn_predict.grid(row=2, column=3)
    
    result_classifier_ext_label = Label(bottom_frame, textvariable=classify_text, width=9)               # label for the result of classifier
    result_classifier_ext_label.grid(row=2, column=4, sticky="W", padx=10, pady=5)
    # -- end: row 2 --
    
    # -- start: row 3 --
    btn_evaluate = Button(bottom_frame, text="Evaluate CNN", command=lambda: model_evaluate("test"))                    # button to evaluate the model by test set
    btn_evaluate.grid(row=3, column=1)
    
    btn_evaluate_ext = Button(bottom_frame, text="Evaluate CNN (extern)", command=lambda: model_evaluate("extern"))       # button to evaluate the model by extern test set
    btn_evaluate_ext.grid(row=3, column=3)
    # -- end: row 3 --
    # ---- end: bottom frame----
    
    # ---- start: error frame (contain the error text if occour an error) ----
    error_frame = Frame(window, width=er_frame_width , height=er_frame_height , bg='grey')
    error_frame.grid(row=4, column=0, padx=er_f_padx , pady=er_f_pady , sticky="nsew")
    error_frame.grid_propagate(False)
    
    error_label = Label(error_frame, textvariable=error_text, bg='grey')
    error_label.grid(row=0, column=0, padx=10, pady=10)
    # ---- end: error frame ----
    
# chose and take the image to visualize and predict, the image is chosen from test set
def btn_load_image():
    global image_to_visualize,index_image_visualized            # global variables references
    img = None                                                  # variable that contain the image to visualize
    index = 0                                                   # index of the chosen img on the test dataset
    error_text.set('')                                          # clean eventual text error
    
    if len(test_image) == 0 or len(test_label) == 0:            # check if there are images in test label
        if len(total_image_ds) != 0:                            # take image from total dataset
            index = random.randint(0,len(total_image_ds)-1)     # chose a random index
            print("Take image to visualize and classify from total_ds, index: ",index)
            index_image_visualized = index      
            img = cv2.imread(total_image_ds[index])             # take the chosen image
            label = str(classes[np.argmax(total_labels_ds[index])])        # take the label of the chosen image    
            label_image_text.set(str(label))                    # shows the label of the chosen image
            label_ext_image_text.set('')                        # clean the label of the image taken from the external test ds
        else:                                                   # no dataset loaded
            error_text.set(er_no_ds_text)                       # shows text error
    else:                                                       # take image from test dataset
        index = random.randint(0,len(test_image)-1)             # chose a random index
        print("Take image to visualize and classify from test set, index: ",index)
        index_image_visualized = index
        img = cv2.imread(test_image[index])                     # take the chosen image
        label = str(classes[np.argmax(test_label[index])])      # take the label of the chosen image 
        label_image_text.set(str(label))                        # shows the label of the chosen image
        label_ext_image_text.set('')                            # clean the label of the image taken from the external test ds
    
    if img is not None:
        dim = ((im_frame_height - 2*im_f_pady) ,(im_frame_height - 2*im_f_pady))    # dim of the image to visualize
        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
        blue,green,red = cv2.split(img)                             # Rearrange colors
        img = cv2.merge((red,green,blue))
        im = PIL.Image.fromarray(np.uint8(img),'RGB')
        image_to_visualize = ImageTk.PhotoImage(image=im)       # update image to visualize in GUI
    
    current_view_to_visualise()                                 # update GUI
    
# chose and take the image to visualize and predict, the image is chosen from external test set
def btn_load_ext_image():
    global image_to_visualize,index_image_visualized            # global variables references
    img = None                                                  # variable that contain the image to visualize
    index = 0                                                   # index of the chosen img on the test dataset
    error_text.set('')                                          # clean eventual text error
    
    if len(test_image_ext) == 0 or len(test_label_ext) == 0:    # check if there are images taken from external test set
        error_text.set(er_no_ext_ds_text)                       # shows text error
    else:                                                       # take image from test dataset
        index = random.randint(0,len(test_image_ext)-1)         # chose a random index
        print("Take image to visualize and classify from external test set, index: ",index)
        index_image_visualized = index
        img = cv2.imread(test_image_ext[index])                 # take the chosen image
        label = str(classes[np.argmax(test_label_ext[index])])  # take the label of the chosen image 
        label_ext_image_text.set(str(label))                    # shows the label of the chosen image
        label_image_text.set('')                                # clean the label of the image taken from the test ds
    
    if img is not None:
        dim = ((im_frame_height - 2*im_f_pady) ,(im_frame_height - 2*im_f_pady))    # dim of the image to visualize
        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)                   # resize the image
        blue,green,red = cv2.split(img)                         # Rearrange colors
        img = cv2.merge((red,green,blue))
        im = PIL.Image.fromarray(np.uint8(img),'RGB')
        image_to_visualize = ImageTk.PhotoImage(image=im)       # update image to visualize in GUI
    
    current_view_to_visualise()                                 # update GUI
    
# ------------------------------------ end: methods for GUI ------------------------------------

# ------------------------------------ start: methods for DS ------------------------------------
# activate a thread to load the ds, in this way the GUI will not be blocked
def btn_load_ds_method(im_width,im_height,im_channel):
    global ds_downloading,ds_analysing          # refer to global variables
    error_text.set('')                          # cleans error text
    if not (ds_downloading or ds_analysing):    # check if program is already downloading the dataset
        ds_downloading = True                   # set control variable
        t = Thread(target=import_image_from_ds, args=(path_dir_ds,im_width,im_height,im_channel,))
        t.start()                               # starts thread for import dataset
    else:
        error_text.set(er_down_ds_text)         # update error text 
        
# method for import the whole dataset, path_ds is the path of the dataset to load. -- P.S. for more detail please read note 0 (at the end of the file)  
def import_image_from_ds(path_ds,im_width,im_height,im_channel):
    global total_image_ds, total_labels_ds,ds_downloading,classes,toggle_duplicate_ds,res_copy_ds_name,img_channel   # refer to global variables
    list_dir_ds = os.listdir(path_ds)               # list of the folders that are in the DS, one folder for each class
    
    # check if the total ds is already loaded
    if len(total_image_ds) != 0:                    # cleans the arrays to allow the dataset to be reloaded
        classes = []
        total_image_ds = []                 
        total_labels_ds = []               

    if not check_image_size_param(im_width,im_height,im_channel): # check parameters of image size
        return  
        
    # check the size of the channel in case of greyscale
    if toggle_grey_scale.get():
        img_channel = 1                 # the number of channel must be 1
        toggle_duplicate_ds.set(True)   # set the var to create the copy of DS

    if toggle_duplicate_ds.get():
        print("Have to duplicate the DS resized.")  
        # check the folder to contain the resized copy of DS
        parent_path = os.path.dirname(path_ds)                  # calculate the parent path
        path_copy = os.path.join(parent_path,res_copy_ds_name)  # create the copy folder in the same folder o the original ds
        if os.path.exists(path_copy):       # check if exist
            shutil.rmtree(path_copy)             # erase all the data in the folder
            print("Copy ds folder present, erase it.")
        else:
            print("Copy ds folder not present.")
        os.makedirs(path_copy)                   # create the folder
        print("Create the copy ds folder.")
    else:
        print("Don't have to duplicate the DS resized.")

    status_DS_text.set('Image DataSet: loading')    # notify the start of the import
    
    print("-------------------- READING DS --------------------")
    print("Read the DS: ",path_ds)                  # status print
    # take the images and labels form DataSet
    for folder in list_dir_ds:                      # for each folder in DS
        print("Read the folder: ",folder)           # status print
        classes.append(str(folder))                 # update classes
        index = classes.index(str(folder))          # take index of classes, is teh label of this class
        p = os.path.join(path_ds,folder)            # path of each folder (of the original folder)
        p_copy = os.path.join(path_copy, folder)    # path of each folder (for the DS copy folder)
        if toggle_duplicate_ds.get():               # resized copy case
            os.makedirs(p_copy, exist_ok=True)          # crea sottocartella nella copia
        # creating a collection with the available images
        for filename in os.listdir(p):                      # for each images on the current folder
            img = cv2.imread(os.path.join(p,filename))      # take current iamge
            if img is not None:                             # check image taken
                # check if is the normal case of the case of the creation of the resized copy DS
                if toggle_duplicate_ds.get():   # resized copy case
                    # check if there is a need to convert in greyscale
                    if toggle_grey_scale.get():
                        img = cv2.imread(percorso_img, cv2.IMREAD_GRAYSCALE)    # convert in greyscale
                    
                    # check if there is a need to resize
                    if img.shape != (img_height, img_width, img_channel):       
                        dim = (img_height, img_width)
                        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                    
                    img = img.astype('float32') / 255                               # normalization
                    output_path = os.path.join(p_copy, filename)    # create the path of the image in the copy DS
                    img_to_save = (img * 255).astype('uint8')       # convert to uint8 for the saving
                    cv2.imwrite(output_path, img_to_save)           # write the image
                    
                    total_image_ds.append(output_path)              # add title of the image to total_image_ds
                    total_labels_ds.append(index)                   # add correlated label to total_lael_ds
                else:                           # normal case
                    total_image_ds.append(os.path.join(p,filename)) # add title of the image to total_image_ds
                    total_labels_ds.append(index)                   # add correlated label to total_lael_ds
            else:
                print("Image loading error...",filename)
    
    # convert in np.array
    total_image_ds = np.array(total_image_ds)
    total_labels_ds = np.array(total_labels_ds)
    # control data print
    print("Num of classes: ",len(classes))
    print("total_image_ds",len(total_image_ds), total_image_ds.shape)
    print("total_labels_ds",len(total_labels_ds), total_labels_ds.shape)
    print("Requied memory for images ds: ",total_image_ds.size * total_image_ds.itemsize / 10**6," MB")
    print("------------------------------------------------------------")
    
    status_DS_text.set('Image DataSet: downloaded')             # notify the end of the process
    error_text.set('')                                          # cleans error text
    ds_downloading = False                                      # reset control variable
    
# activate a thread to load the extern test ds, in this way the GUI will not be blocked
def btn_load_ext_ds_method(im_width,im_height,im_channel):
    global ext_ds_downloading,ds_analysing      # refer to global variables
    error_text.set('')                          # cleans error text
    if not (ext_ds_downloading or ds_analysing):# check if program is already downloading the dataset
        ext_ds_downloading = True               # set control variable
        t = Thread(target=import_image_from_ext_test_ds, args=(path_dir_test_ds,im_width,im_height,im_channel))
        t.start()                               # starts thread for import dataset
    else:
        error_text.set(er_down_ds_text)         # update error text 
    
# method for import the whole test dataset, path_ds is the path of the dataset to load. -- P.S. same detail of the 'import_image_from_ds' method in Note 0
def import_image_from_ext_test_ds(path_ds,im_width,im_height,im_channel):
    global test_image_ext, test_label_ext,ext_ds_downloading    # refer to global variables
    image_ds = []
    labels_ds = []                                      # local variables
    list_dir_ds = os.listdir(path_ds)                   # list of the folders that are in the DS, one folder for each class
    error_text.set('')                                  # clear error text
    
    # check if the total ds is already loaded, it's necessary for a correct formatting of the labels
    if len(total_image_ds) == 0:
        error_text.set(er_no_ds_text)                   # update error text
        return
        
    if not check_image_size_param(im_width,im_height,im_channel): # check parameters of image size
        return  
    
    status_ext_test_DS_text.set('External test DS: loading')   # notify the start of the import
    # take the images and labels form DataSet
    for folder in list_dir_ds:                      # for each folder in DS
        print("Read the folder: ",folder)           # status print
        index = classes.index(str(folder))          # take index of classes, is teh label of this class
        p = os.path.join(path_ds,folder)            # path of each folder
        #creating a collection with the available images
        for filename in os.listdir(p):                      # for each images on the current folder
            img = cv2.imread(os.path.join(p,filename))      # take current iamge
            if img is not None:                             # check image taken
                image_ds.append(os.path.join(p,filename)) # add title of the image to total_image_ds
                labels_ds.append(index)               # add correlated label to total_lael_ds
            else:
                print("Image loading error...",filename)
                    
    # convert in np.array
    test_image_ext = np.array(image_ds)
    test_label_ext = np.array(labels_ds)
    # control data print
    print("test_image_ext",len(test_image_ext), test_image_ext.shape)
    print("test_label_ext",len(test_label_ext), test_label_ext.shape)
    print("Requied memory for images ds: ",test_image_ext.size * test_image_ext.itemsize / 10**9," GB")
    
    status_ext_test_DS_text.set('Extern test DS: downloaded')       # notify the end of the process
    error_text.set('')                                              # cleans error text
    ext_ds_downloading = False                                      # reset control variable
    
# activate a thread to load the ds, in this way the GUI will not be blocked
def btn_analyse_ds(folder_name,im_width,im_height,im_channel):
    global ds_downloading,ds_analysing          # refer to global variables
    error_text.set('')                          # cleans error text
    if not (ds_downloading or ds_analysing):    # check if program is already downloading the dataset
        ds_analysing = True                     # set control variable
        t = Thread(target=analyse_ds, args=(folder_name,im_width,im_height,im_channel,))
        t.start()                               # starts thread for import dataset
        check_thread(t, window)                 # controlla periodicamente se ha finito
    else:
        error_text.set(er_down_ds_text)         # update error text 
     
# method to check if the thread for analysing DS is done or not     
def check_thread(thread, root):
    if thread.is_alive():
        root.after(100, lambda: check_thread(thread, root))   # ricontrolla dopo 100 ms
    else:
        show_plot_analysing()                           # funzione che vuoi eseguire nel thread principale
        status_DS_text.set('Image DataSet: analysed')   # notify the end of the analysing

def show_plot_analysing():
    # show a plot that displays BRG bar gram for each class
    for k,v in RGB_mean_dict.items():                                   # for to slide the sorted dict
        color = ["Blue", "Green", "Red"]                                # laber for bar in the bar gram                        
        x_pos = np.arange(len(color))
        plt.bar(x_pos, v, align='center')
        plt.xticks(x_pos, color)
        plt.ylabel('Value')
        plt.xlabel('Color')
        plt.title(k)                                                    # the title of the plot is the class 
        plt.show()                                                      # shows bar grams
       
# method for import the whole test dataset, new_path_ds is the path of the dataset to load. -- P.S. same detail of the 'import_image_from_ds' method in Note 0
def analyse_ds(folder_name,im_width,im_height,im_channel):
    global ds_analysing, RGB_mean_dict
    # variables
    img_number = 0                                  # total number of images in the dataset
    classes = {}                                    # dictionary containing all the classes in the dataset and the number of images of each class
    format_dict = {}                                # dictionary containing all the image formats in the dataset and for each of them the number of images
    shape_dict = {}                                 # dictionary containing all the image shapes in the dataset and for each of them the number of images
    top_shape_images = 10                           # the top frequent shapes for the images in the dataset
    corrupt_images = []                             # array that will contain the name of the corrupted images that are in the dataset
    
    new_path_ds = path_dir_ds   
    
    # take the list of the folders that are in the DS, one folder for each class
    if is_valid_folder_name(folder_name) and os.path.isdir(folder): # check if the user write a valide name for the DS and there is a folder
        new_path_ds = os.path.join("Dataset",folder_name)
        list_dir_ds = os.listdir(new_path_ds) 
    else:                                           # take default ds path
        new_path_ds = path_dir_ds
        list_dir_ds = os.listdir(path_dir_ds)    

    if not check_image_size_param(im_width,im_height,im_channel): # check parameters of image size
        return  
    
    status_DS_text.set('Image DataSet: analysing')  # notify the start of the analysing
    print("-------------------- ANALYSING DS --------------------")
    print(" -- Start time analyse the DS: ", path_dir_ds, " --")
    print("Image size for resize is: (",img_width,",",img_height,",",img_channel,")")
    im_ds = []                  # tensor that contain the images of one batch from the set
    lab_ds = []                 # tensor that contain the labels of one batch from the set
    im_tens = []                # tensor that contain the residual images (case where size/batch_size has a rest) from the set                                 
    
    # time variables
    resize_time = 0                                 # set the time needed to resize all the images in the DS
    save_path_time = 0                              # set the time needed to save the path for all the images in the DS
    tens_time = 0                                   # set the time needed to convert in tensor and add to list for all the images in the DS
    
    start_time = time.time()                        # start time for analysing read and resize DS
    for folder in list_dir_ds:                      # for each folder in DS
        print("Analyse the folder: ",folder)        # status print
        p = os.path.join(new_path_ds,folder)        # path of each folder
        for filename in os.listdir(p):              # analyse all images
            img = cv2.imread(os.path.join(p,filename))  # take current iamge
            if img is not None:                         # check image taken
                temp_start = time.time()
                im_ds.append(os.path.join(p,filename))  # add title of the image to total_image_ds
                temp_end = time.time()
                save_path_time += (temp_end - temp_start)
                
                #check if the image is in the correct shape for the CNN (shape specified in the global variables)
                if img.shape != (img_width, img_height, img_channel):    
                    temp_start = time.time()
                    dim = (img_width ,img_height)
                    img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)       # resize the image
                    temp_end = time.time()
                    resize_time += (temp_end - temp_start)
                    
                    temp_start = time.time()
                    img = img.astype('float32') / 255                               # normalization
                    im_tens.append(tf.convert_to_tensor(img, dtype=tf.float32))     # add new element and convert to TF tensors
                    temp_end = time.time()
                    tens_time += (temp_end - temp_start)
                else:
                    temp_start = time.time()
                    img = img.astype('float32') / 255                               # normalization
                    im_tens.append(tf.convert_to_tensor(img, dtype=tf.float32))     # add new element and convert to TF tensors
                    temp_end = time.time()
                    tens_time += (temp_end - temp_start)
            else:
                 print("Image loading error in generator_train...",img_path)
                 
    end_time = time.time()                          # end time for analysing read and resize DS
    total_time = end_time - start_time
    print(f"Time to read and resize the images of DS: ", converti_secondi(total_time))  # print time
    percent = (resize_time / total_time) * 100
    print(f"Time to resize the images of DS: ", converti_secondi(resize_time),f" {percent:.2f}% of total time.")                        # print time
    percent = (save_path_time / total_time) * 100
    print(f"Time to save the path of the images of DS: ", converti_secondi(save_path_time),f" {percent:.2f}% of total time.")           # print time
    percent = (tens_time / total_time) * 100
    print(f"Time to convert and add to the tensor of the images of DS: ", converti_secondi(tens_time),f" {percent:.2f}% of total time.")# print time
    
    print("\n -- Start analyse the DS: ", path_dir_ds, " --")
    # slide the images and labels form DataSet
    for folder in list_dir_ds:                      # for each folder in DS
        print("Analyse the folder: ",folder)        # status print
        if classes.get(str(folder)) is None:        # check if the classes is already registered
            classes[str(folder)] = 0                # set counter equal 0
        
        p = os.path.join(new_path_ds,folder)            # path of each folder
        
        for filename in os.listdir(p):                      # for each images on the current folder
            img_number +=1                                  # update images counter
            classes[str(folder)] += 1                       # update images in classes counter
            # I get the image format, I assume that there are no dots in the name of the images
            split_name = str(filename).split('.')           # split the filename
            format_name = split_name[-1]                    # get the format
            # check format
            if format_dict.get(format_name) is None:     # check if the format is already registered
                format_dict[format_name] = 1             # set counter equal 0
            else:
                format_dict[format_name] += 1            # update counter    
            
            img = cv2.imread(os.path.join(p,filename))      # take current iamge
            if img is not None:                             # check image taken
                # check shape
                if shape_dict.get(str(img.shape)) is None:     # check if the shape is already registered
                    shape_dict[str(img.shape)] = 1             # set counter equal 0
                else:
                    shape_dict[str(img.shape)] += 1            # update counter  
                # see the color contribution
                # for each image calculate the mean of the value of the Red, Green and Blue
                red_mean = img[:,:,2].mean()
                green_mean = img[:,:,1].mean()
                blue_mean = img[:,:,0].mean()
                if RGB_mean_dict.get(str(folder)) is None:                              # check if there are already value for this class
                    RGB_mean_dict[str(folder)] = [blue_mean,green_mean,red_mean]        # set with value
                else:
                    temp_RGB = RGB_mean_dict[str(folder)]
                    new_RGB = [temp_RGB[0] + blue_mean, temp_RGB[1] + green_mean , temp_RGB[2] + red_mean ]
                    RGB_mean_dict[str(folder)] = new_RGB                                # update value
            else:                                           # corrupted image
                corrupt_images.append(str(filename))        # update 
                if del_corrupt_img:                         # check if the porgram has to delete the corrupted images or not
                    os.remove(os.path.join(p,filename))     # delete the images

    # Dataset characteristics output
    print("The dataset characteristics are:")
    print("- The dataset has " , img_number, " images.")                                      # total number of images in dataset
    print("- The number of classes are ",len(classes.keys()), ". ", list(classes.keys()))     # number of classes and name of each class
    
    print("List of the classes and number of related images for each one:")
    sorted_class_dict = dict(sorted(classes.items(), key=lambda x: x[1], reverse=True))     # sort the classes dictionary. reverse = true the biggest class will be the first                                                                            # initialize the counter
    for k,v in sorted_class_dict.items():                                                   # for to slide the sorted dict
        print('- ', k, ': ', v)                                                                   # print key and value       

    if len(corrupt_images) != 0:                                                            #check if there are corrupted images
        print("There are ", len(corrupt_images)," corrupted images.\n",corrupt_images)

    print("The number of different images shapes are ",len(shape_dict.keys()))              # number of shapes and images number for each shape
    
    # for to see the top k shape image with k as top_shape_images
    sorted_shape_dict = dict(sorted(shape_dict.items(), key=lambda x: x[1], reverse=True))  # sort the shape_dictionary. reverse = true the most used shape will be the first
    count = 0                                                                               # initialize the counter
    for k,v in sorted_shape_dict.items():                                                   # for to slide the sorted dict
        if count >= top_shape_images:                                                       # check if is out of band for top k
            break
        print('- ', k, ': ', v)                                                                   # print key and value
        count +=1                                                                           # update counter   

    print("The number of different images format are ",len(format_dict.keys()))             # number of format and images number for each format
    # for to see the top k format image with k as top_shape_images
    sorted_format_dict = dict(sorted(format_dict.items(), key=lambda x: x[1], reverse=True))# sort the shape_dictionary. reverse = true the most used shape will be the first
    count = 0                                                                               # initialize the counter
    for k,v in sorted_format_dict.items():                                                  # for to slide the sorted dict
        if count >= top_shape_images:                                                       # check if is out of band for top k
            break
        print('- ', k, ': ', v)                                                                   # print key and value
        count +=1                                                                           # update counter  
        
    ds_analysing = False                # reset the control var
    print("------------------------------------------------------------")

# method for preprocessing and split the dataset
def make_set_ds():
    global test_image, test_label, train_image, train_label, val_img,val_label          # global variables references
    seed = random.randint(1, 42)            # calculate the seed
    
    # ----- preprocessing and reshape ----
    image_ds = np.array(total_image_ds)
    labels_ds = to_categorical(total_labels_ds,num_classes=len(classes))    # transform label in categorical format
    
    # ---- generete the training and test set ----
    train_img_temp, test_image, train_label_temp, test_label = train_test_split(image_ds, labels_ds, test_size=test_set_split , random_state=seed, shuffle=True)    # split to generate train and test set
    train_image, val_img, train_label, val_label = train_test_split(train_img_temp, train_label_temp, test_size=test_set_split , random_state=seed, shuffle=True)   # split to generate validation set from train set
    
    # information print
    print("-------------------- SETS --------------------")
    print("Make this sets:")
    print("train_image len:",len(train_image), train_image.shape)
    print("train_label len",len(train_label), train_label.shape)
    print("val_img len:",len(val_img), val_img.shape)
    print("val_label len",len(val_label), val_label.shape)
    print("test_image len:",len(test_image), test_image.shape)
    print("test_label len",len(test_label), test_label.shape)
    print("------------------------------------------------------------")

# ------------------------------------ end: methods for DS ------------------------------------

# ------------------ start: generetor function ------------------
# explanation: for large dataset with large image or big batch size the memory memory may not be sufficient. 
#              To avoid memory overflow, the sets are supplied in batches via yeld istruction.
# define generator function to do the training set
def generator_train():
    # create the tensor that will contain the data
    img_tensor = []                                             # tensor that contain the images of one batch from the set
    label_tensor = []                                           # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                        # check if it has to truncate or not the set
        rest = batch_size - (len(train_image) % batch_size)     # check if the division by batch_size produce rest
    else:
        rest = batch_size                                       # set always truncated
    
    for idx in range(len(train_image)):                         # organize the sample in batch
        # take one image and the corresponding labels
        img_path = train_image[idx]                              
        label = train_label[idx]
        # check if the DS is in greyscale
        if toggle_grey_scale.get():
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # take current image (greyscale)
        else:
            img = cv2.imread(img_path)                          # take current iamge (RGB)
        if img is not None:                             # check image taken
            #check if the image is in the correct shape for the CNN (shape specified in the global variables)
            if img.shape != (img_width, img_height, img_channel):       
                dim = (img_height ,img_width)
                img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)        # resize the image
                img = img.astype('float32') / 255                                # normalization
                img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))   # add new element and convert to TF tensors
            else:
                img = img.astype('float32') / 255                                       # normalization
                img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))          # add new element and convert to TF tensors
        else:
             print("Image loading error in analysing DS...",img_path)
        
        # add new element and convert to TF tensors
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future (sample in the rest)
            img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(img_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield img_tensor, label_tensor                      # return the batch
            # clean list
            img_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(train_image) - 1):                       # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    img_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])

                yield img_tensor, label_tensor                  # return the last batch
                

# define generator function to do the validation set
def generator_val():
    # create the tensor that will contain the data
    img_tensor = []                                             # tensor that contain the images of one batch from the set
    label_tensor = []                                           # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                        # check if it has to truncate or not the set
        rest = batch_size - (len(val_img) % batch_size)         # check if the division by batch_size produce rest
    else:
        rest = batch_size                                       # set always truncated
    
    for idx in range(len(val_img)):                             # organize the sample in batch
        # take one image and the corresponding mask
        img_path = val_img[idx]
        label = val_label[idx]
        # check if the DS is in greyscale
        if toggle_grey_scale.get():
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # take current image (greyscale)
        else:
            img = cv2.imread(img_path)                          # take current iamge (RGB)
        if img is not None:                             # check image taken
            #check if the image is in the correct shape for the CNN (shape specified in the global variables)
            if img.shape != (img_width, img_height, img_channel):       
                dim = (img_height ,img_width)
                img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)        # resize the image
                img = img.astype('float32') / 255                         # normalization
                img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))   # add new element and convert to TF tensors
            else:
                img = img.astype('float32') / 255                                       # normalization
                img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))          # add new element and convert to TF tensors
        else:
            print("Image loading error in generator_val...",img_path)
        
        # add new element and convert to TF tensors
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future
            img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(img_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield img_tensor, label_tensor                      # return the batch
            # clean list
            img_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(val_img) - 1):                           # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    img_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                yield img_tensor, label_tensor                  # return the last batch
        
# define generator function to do the test set
def generator_test():
    # create the tensor that will contain the data
    img_tensor = []                                             # tensor that contain the images of one batch from the set
    label_tensor = []                                           # tensor that contain the labels of one batch from the set
    img_rest_tensor = []                                        # tensor that contain the residual images (case where size/batch_size has a rest) from the set
    label_rest_tensor = []                                      # tensor that contain the residual labels (case where size/batch_size has a rest) from the set
    
    if not truncate_set:                                        # check if it has to truncate or not the set
        rest = batch_size - (len(test_image) % batch_size)      # check if the division by batch_size produce rest
    else:
        rest = batch_size                                       # set always truncated
    
    for idx in range(len(test_image)):                          # organize the sample in batch
        # extract one image and the corresponding mask
        img_path = test_image[idx]
        label = test_label[idx]
        # check if the DS is in greyscale
        if toggle_grey_scale.get():
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)    # take current image (greyscale)
        else:
            img = cv2.imread(img_path)                          # take current iamge (RGB)
        if img is not None:                             # check image taken
            #check if the image is in the correct shape for the CNN (shape specified in the global variables)
            if img.shape != (img_width, img_height, img_channel):       
                dim = (img_height ,img_width)
                img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)        # resize the image
                img = img.astype('float32') / 255                         # normalization
                img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))   # add new element and convert to TF tensors
            else:
                img = img.astype('float32') / 255                                       # normalization
                img_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))          # add new element and convert to TF tensors
        else:
            print("Image loading error in generator_test...",img_path)

        # add new element and convert to TF tensors
        label_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))
        
        if rest != batch_size and idx < rest:                   #check for the rest
            # add this sample for the future
            img_rest_tensor.append(tf.convert_to_tensor(img, dtype=tf.float32))
            label_rest_tensor.append(tf.convert_to_tensor(label, dtype=tf.float32))

        if len(img_tensor) == batch_size:                       # check to see if batch is full (reached batch_size)
            yield img_tensor, label_tensor                      # return the batch
            # clean list
            img_tensor.clear()
            label_tensor.clear()
            
        if idx == (len(test_image) - 1):                        # check if the set is finished, last batch
            if rest != batch_size:                              # check if there are rest to fix
                #there are samples that don't complete a batch, add rest sample to complete the last batch
                for i in range(rest):
                    img_tensor.append(img_rest_tensor[i])
                    label_tensor.append(label_rest_tensor[i])
                yield img_tensor, label_tensor                  # return the last batch

# ------------------ end: generetor function ------------------
# ------------------------------------ end: methods for DS ------------------------------------

# ------------------------------------ start: methods for CNN model ------------------------------------
# method for check the image size parameter for the input layer
def check_image_size_param(height,width,channel):
    global img_height, img_width, img_channel       # global variables references

    if height:                                      # control check for img_height 
        if height.isnumeric():                      # check if the string is a number or not, this method doesn't recongnize the negative number but it's okay for our case, img_height must be positive number
            try:
                int_img_height = int(height)        # convert to int if is possible
            except:
                error_text.set(er_format_image_size_text)    # update error text  
                return False
            img_height = int_img_height             # upate the value of img_height
        else:                                           
            error_text.set(er_format_image_size_text)        # update error text 
            return False
    # if user don't insert a value for img_width the program will use a default value
    
    if width:                                       # control check for img_width 
        if width.isnumeric():                       # check if the string is a number or not, this method doesn't recongnize the negative number but it's okay for our case, img_width must be positive number
            try:
                int_img_width = int(width)          # convert to int if is possible
            except:
                error_text.set(er_format_image_size_text)    # update error text  
                return False
            img_width = int_img_width               # upate the value of img_width
        else:                                           
            error_text.set(er_format_image_size_text)        # update error text 
            return False
    # if user don't insert a value for img_width the program will use a default value
    
    if channel:                                     # control check for img_channel 
        if channel.isnumeric():                     # check if the string is a number or not, this method doesn't recongnize the negative number but it's okay for our case, img_channel must be positive number
            try:
                int_img_channel = int(channel)      # convert to int if is possible
            except:
                error_text.set(er_format_image_size_text)    # update error text  
                return False
            img_channel = int_img_channel           # upate the value of img_channel
        else:                                           
            error_text.set(er_format_image_size_text)        # update error text 
            return False
    # if user don't insert a value for img_channel the program will use a default value
    
    return True                                         # all parameter are ok
    
def is_valid_folder_name(name):
    # Controlla se il nome è vuoto o solo spazi
    if not name.strip():
        return False

    # Caratteri vietati per Windows
    invalid_chars = '<>:"/\\|?*'
    if any(char in name for char in invalid_chars):
        return False

    # Riservati su Windows (opzionale)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'COM{i}' for i in range(1, 10)} | {f'LPT{i}' for i in range(1, 10)}
    if name.upper() in reserved:
        return False

    return True

# method for check the fit parameter insert by user. return 'True' whether the parameter are correct, 'False' is the parameter aren't correct
def check_fit_param(number_epoch,num_batch_size,num_early_patience):
    global epochs, batch_size, early_patience           # global variables references

    if number_epoch:                                    # control check for number of epochs of the train   
        if number_epoch.isnumeric():                    # check if the string is a number or not, this method doesn't recongnize the negative number but it's okay for our case, number of epochs must be positive number
            try:
                int_numb_epoch = int(number_epoch)      # convert to int if is possible
            except:
                error_text.set(er_format_epoch_text)    # update error text  
                return False
            epochs = int_numb_epoch                     # upate the value of batch_size
        else:                                           
            error_text.set(er_format_epoch_text)        # update error text 
            return False
    # if user don't insert a value for the number of epochs the program will use a default value

    if num_batch_size:                                    # control check for batch size   
        if num_batch_size.isnumeric():                    # check if the string is a number or not, this method doesn't recongnize the negative number but it's okay for our case, number of epochs must be positive number
            try:
                int_num_batch_size = int(num_batch_size)  # convert to int if is possible
            except:
                error_text.set(er_format_batch_size_text) # update error text  
                return False
            batch_size = int_num_batch_size               # upate the value of epochs
        else:                                           
            error_text.set(er_format_batch_size_text)     # update error text 
            return False
    # if user don't insert a value for the batch_size the program will use a default value

    if num_early_patience:                                    # control check for batch size   
        if num_early_patience.isnumeric():                    # check if the string is a number or not, this method doesn't recongnize the negative number but it's okay for our case, number of epochs must be positive number
            try:
                int_num_early_patience = int(num_early_patience)  # convert to int if is possible
            except:
                error_text.set(er_format_early_text)          # update error text  
                return False
            early_patience = int_num_early_patience           # upate the value of epochs
        else:                                           
            error_text.set(er_format_early_text)              # update error text 
            return False
    # if user don't insert a value for the early patience the program will use a default value

    return True                                         # all parameter are ok
    
# method to check if the number of test passed as parameter is correct or not
def validate_num_test(value):
    try:
        num = int(value)                # try to convert in int
        if num > 0:
            return num                  # correct number
        else:
            return 1                    # incorrect number -> return 1
    except (ValueError, TypeError):
        return 1                        # incorrect number or error -> return 1

# method to create and fit model, 'chosen_model' indicate the model chosen by user by the menu tillbar
# 'number_epoch' is number of epoch insert by user , 'num_batch_size' is the batch size insert by user 
# 'early_patience' is the patience for early stopping insert b user
def make_fit_model(chosen_model,number_epoch,num_batch_size,num_early_patience,num_test):
    global model_trained, test_image, test_label, train_image, train_label, network, epochs, batch_size, early_patience, past_param_model  # global variables references
    make_model = True                                   # var that indicate if this call of method has to make and fit CNN model
    n_test = 1                                          # the number of fit to do
    # Dictionaries for collecting metrics
    history_train = {'accuracy': [], 'loss': []}        # to contain value for the training set
    history_val   = {'accuracy': [], 'loss': []}        # to contain value for the validation set
    history_test  = {'accuracy': [], 'loss': []}        # to contain value for the test set

    error_text.set('')                                  # cleans error text
    if len(total_image_ds) == 0:                        # control check
        error_text.set(er_train_without_ds_text)        # update error text
        return
    
    if not check_fit_param(number_epoch,num_batch_size,num_early_patience): # check parameters
        return                                          # at least one parameter isn't correct

    n_test = validate_num_test(num_test)                # check the number of test to do
    print("------------------------ MAKE AND FIT MODEL ------------------------")
    start_all_time = time.time()                        # start time for all training
    for i in range(n_test):
        print("-------- start: test num ",i," --------")
        
        make_set_ds()                                   # split and create the sets
        status_model_text.set('Model: working')         # notify the start of the process
        # ---- make the model -----
        print("------------------------ make model ------------------------")
        if not model_trained:                   # check if there is a ready model or not 
            if chosen_model == "None":                          # check if the CNN model has been selected by user
                error_text.set(er_no_model_specified_text)      # update error text
                return                                          # user must specify a template
            else:
                print("First time, making model...")
                # save the param for next call of this method
                past_param_model["type_model"] = chosen_model         # save type of the model
                past_param_model["epochs"] = epochs                   # save type of the model
                past_param_model["batch_size"] = batch_size           # save type of the model
                past_param_model["early_patience"] = early_patience   # save type of the model

                make_model = True                               # has to make model, set variable
        else:                                   # there is a past model, verify if the parameters are the same of this call
            print("model already done, checking...")
            make_model = not check_past_model(chosen_model)           # set variable

        print("Check past: ", check_past_model(chosen_model), " Make model: ",make_model)
        if make_model:                          # make and fit another model
            # the actual parameters are different respect to the previous ones. Save the param for next call of this method
            past_param_model["type_model"] = chosen_model         # save type of the model
            past_param_model["epochs"] = epochs                   # save type of the model
            past_param_model["batch_size"] = batch_size           # save type of the model
            past_param_model["early_patience"] = early_patience   # save type of the model
            # check what type of model the user want to make and fit, user can chose form this option: 'None','AlexNet','GoogleNet','Ifrit'
            if chosen_model == "None":
                error_text.set(er_no_model_specified_text)      # update error text
                return                                          # user must specify a template
            elif chosen_model == "AlexNet":
                ANet_Model = ANet.AlexNet(len(classes),img_width,img_height,img_channel)    # create an instance of the AlexNet class
                ANet_Model.make_model()                         # make model (AlexNet architecture)
                ANet_Model.compile_model()                      # compile 
                network = ANet_Model.return_model()             # return model
            elif chosen_model == "GoogleNet":
                GLNet_Model = GLNet.GoogLeNet(len(classes),img_width,img_height,img_channel)     # create an instance of the AlexNet class
                GLNet_Model.make_model()                        # make model (GoogLeNet architecture)
                GLNet_Model.compile_model()                     # compile model
                network = GLNet_Model.return_model()            # return model
                return
            elif chosen_model == "Ifrit_1":                    
                Ifrit_Model = IfritNet.IfritNet(len(classes),img_width,img_height,img_channel)    # create an instance of the IfriNet class
                Ifrit_Model.make_model(1)                       # make model (IfriNet 1 architecture)
                Ifrit_Model.compile_model()                     # compile model
                network = Ifrit_Model.return_model()            # return model
            elif chosen_model == "Ifrit_2":                       
                Ifrit_Model = IfritNet.IfritNet(len(classes),img_width,img_height,img_channel)    # create an instance of the IfriNet class
                Ifrit_Model.make_model(2)                       # make model (IfriNet 1 architecture)
                Ifrit_Model.compile_model()                     # compile model
                network = Ifrit_Model.return_model()            # return model
            elif chosen_model == "Ifrit_3":                       
                Ifrit_Model = IfritNet.IfritNet(len(classes),img_width,img_height,img_channel)    # create an instance of the IfriNet class
                Ifrit_Model.make_model(3)                       # make model (IfriNet 1 architecture)
                Ifrit_Model.compile_model()                     # compile model
                network = Ifrit_Model.return_model()            # return model
            elif chosen_model == "Ifrit_4":                       
                Ifrit_Model = IfritNet.IfritNet(len(classes),img_width,img_height,img_channel)    # create an instance of the IfriNet class
                Ifrit_Model.make_model(4)                       # make model (IfriNet 1 architecture)
                Ifrit_Model.compile_model()                     # compile model
                network = Ifrit_Model.return_model()            # return model
        
        # create TRAIN SET using generator function and specifying shapes and dtypes
        train_set = tf.data.Dataset.from_generator(generator_train, 
                                                 output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32))).repeat()
        
        # create VALIDATION SET using generator function and specifying shapes and dtypes
        val_set = tf.data.Dataset.from_generator(generator_val, 
                                                 output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32))).repeat()
        # create TEST SET using generator function and specifying shapes and dtypes
        test_set = tf.data.Dataset.from_generator(generator_test, 
                                                 output_signature=(tf.TensorSpec(shape=(batch_size ,img_width , img_height , img_channel), dtype=tf.float32),
                                                                   tf.TensorSpec(shape=(batch_size, len(classes)), dtype=tf.float32))).repeat()

        # ---- fit the model -----
        print("------------------------ fit model ------------------------")
        gpu_memory_usage()                                  # check print
        
        checkpoint = ModelCheckpoint(filepath = path_check_point_model+'/weight_seg_'+chosen_model+".keras", verbose = 1, save_best_only = True, monitor='val_loss', mode='min') # val_loss, min, val_categorical_accuracy, max
        eStop = EarlyStopping(patience = early_patience, verbose = 1, restore_best_weights = True, monitor='val_loss')
        
        # -- calculate steps for the sets --
        # train steps
        if (len(train_image) % batch_size) == 0 or truncate_set:        # check if the division by batch_size produce rest
            train_step = int(math.floor(len(train_image) / batch_size))
        else:
            train_step = int(math.floor((len(train_image) / batch_size)) + 1)
        
        # val steps
        if (len(val_img) % batch_size) == 0 or truncate_set:            # check if the division by batch_size produce rest
            val_step = int(math.floor(len(val_img) / batch_size))
        else:
            val_step = int(math.floor((len(val_img) / batch_size)) + 1)
            
        # test steps
        if (len(test_image) % batch_size) == 0 or truncate_set:         # check if the division by batch_size produce rest
            test_step = int(math.floor(len(test_image) / batch_size))
        else:
            test_step = int(math.floor((len(test_image) / batch_size)) + 1)

        start_time = time.time()                            # start time for training
        history = network.fit(train_set,validation_data=val_set,steps_per_epoch=train_step,validation_steps=val_step, epochs=epochs, callbacks = [checkpoint, eStop])     # fit model
        end_time = time.time()                              # end time for training
        print(f"Time for training the model: ", converti_secondi(end_time - start_time)," - of the test number: ",i)  # print time to train the model
        
        model_trained = True                                # update status variable
        status_model_text.set('Model: trained')             # notify the end of the process
        
        if n_test == 1:             # case only 1 fit
            plot_fit_result(history.history,0)                  # visualize the value for the fit - history.history is a dictionary - call method for plot train result
        else:                       # case more fit, statistical pourpose
            history_train['accuracy'].append(history.history['accuracy'][-1])
            history_train['loss'].append(history.history['loss'][-1])
            history_val['accuracy'].append(history.history['val_accuracy'][-1])
            history_val['loss'].append(history.history['val_loss'][-1])
            
        
        # -- evaluate on test set and make plots --
        test_loss, test_acc = network.evaluate(test_set, steps=test_step)       # obtain loss and accuracy metrics
        if n_test == 1:             # case only 1 fit
            dict_metrics = {'loss': test_loss, 'accuracy': test_acc}            # create a dictionary contain the metrics
            plot_fit_result(dict_metrics,1)                                     # plot the values obtained
            compute_confusion_matrix(network, test_set, test_step, classes)     # call method to obtain the confusion matrix    
        else:                       # case more fit, statistical pourpose
            history_test['loss'].append(test_loss)
            history_test['accuracy'].append(test_acc)
        
        print("-------- end: test num ",i," --------")
    end_all_time = time.time()                              # end time for all training
    
    # plot and calculate the mean in case of number of fit test done
    if n_test > 1:
        plot_accuracy_and_loss(history_train, history_val, history_test)
        print_average_metrics(history_train, history_val, history_test)
    
    print(f"Time for training all the tests: ", converti_secondi(end_all_time - start_all_time))
    print("------------------------------------------------")
    
# method for evaluate the model by the test set. 'param' specify if the evaluate hase to use test set or external test set ('',)
def model_evaluate(param):
    data_test = []
    labels_test = []
    
    error_text.set('')                                  # clear error text
    if not model_trained:                               # check control if there is a CNN model ready
        error_text.set(er_eval_without_model_text)      # update error text
        return
    if param == "test":                                 # use test_set
        # check if there are images in test label didn't use in the train, ( test_image , test_label)
        if len(test_image) == 0 or len(test_label) == 0:
            # split
            data_train, data_test, labels_train, labels_test = train_test_split(total_image_ds, total_labels_ds, test_size=test_set_split , random_state=42)
            for img_path in data_test:
                img = cv2.imread(img_path)              # take current iamge
                if img is not None:                     # check image taken
                #check if the image is in the correct shape for the CNN (shape specified in the global variables)
                    if img.shape != (img_width, img_height, img_channel):       
                        dim = (img_height ,img_width)
                        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                        img = img.astype('float32') / 255                           # normalization
                        data_test.append(img)                                       # add new element
                    else:
                        img = img.astype('float32') / 255                           # normalization
                        data_test.append(img)                                       # add new element
                else:
                    print("Image loading error in model_evaluate...",img_path)
            
            labels_test = to_categorical(labels_test,num_classes=len(classes))      # transform label in categorical format
        else:
            # take all the images from th test set
            for img_path in test_image:
                img = cv2.imread(img_path)              # take current iamge
                if img is not None:                     # check image taken
                #check if the image is in the correct shape for the CNN (shape specified in the global variables)
                    if img.shape != (img_width, img_height, img_channel):       
                        dim = (img_height ,img_width)
                        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                        img = img.astype('float32') / 255                           # normalization
                        data_test.append(img)                                       # add new element
                    else:
                        img = img.astype('float32') / 255                           # normalization
                        data_test.append(img)                                       # add new element
                else:
                    print("Image loading error in model_evaluate...",img_path)
   
            labels_test = test_label                                # take labels 
    elif param == "extern":                                         # use extern test ds
        if len(test_image_ext) == 0 or len(test_label_ext) == 0:    # check if the external test ds has already loaded
            error_text.set(er_eval_without_model_text)              # update error text
            return
        else:
            data_test = test_image_ext.astype('float32') / 255                              # normalization
            labels_test = to_categorical(test_label_ext,num_classes=len(classes))           # transform label in categorical format

    # resize the set of the image
    data_test = np.array(data_test)
    data_test = data_test.reshape((len(data_test), img_width, img_height, img_channel))     
    
    test_loss, test_acc = network.evaluate(data_test, labels_test)                      # obtain loss and accuracy metrics
    dict_metrics = {'loss': test_loss, 'accuracy': test_acc}                            # create a dictionary contain the metrics
    plot_fit_result(dict_metrics,1)                                                     # plot the values obtained
    

# check if there is a saved model and load it
def load_saved_model(model_name):
    global network,model_trained                                # reference to a global variables
    error_text.set('')                                          # clean eventual text error
    if model_name:                                              # check if user has entered a model name
        save_path = os.path.join(path_dir_model,model_name)
        if os.path.exists(save_path):                           # check if there is a model
            network = load_model(save_path)                     # load model
            model_trained = True                                # update status variable
            status_model_text.set('Model: trained')
        else:
            error_text.set(er_load_model_unknown_text)          # shows text error
    else:
        error_text.set(er_load_model_text)                      # shows text error
        
# save model
def save_model(model_name):
    global network,model_trained                                # reference to a global variables
    error_text.set('')                                          # clean eventual text error
    if model_name:                                              # check if user has entered a model name
        save_path = os.path.join(path_dir_model,model_name)
        network.save(save_path)                                 # save the trained model, creates a HDF5 file
    else:
        error_text.set(er_save_model_text)                      # shows text error
        
# method to predict the label associated to the image visualized, the image can be taken from external test set or not (distinction made through the text value of the labels associated with the image)
def predict(): 
    if index_image_visualized != -1 and model_trained:          # control check for the image visualized
        error_text.set('')                                      # clean the error_text
        # control check for origin of the image displayed
        if label_image_text.get() != '':                        # image from internal test set
            if len(test_image) == 0 or len(test_label) == 0:    # check to know what set is used (total_image_ds or total_test_image)
                print("Take image from total_image_ds...")      # take from total_image_ds
                
                # take image
                img = cv2.imread(total_image_ds[index_image_visualized])            # take current iamge
                if img is not None:                     # check image taken
                    #check if the image is in the correct shape for the CNN (shape specified in the global variables)
                    if img.shape != (img_width, img_height, img_channel):       
                        dim = (img_height ,img_width)
                        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                        img = img.astype('float32') / 255                           # normalization
                    else:
                        img = img.astype('float32') / 255                           # normalization
                else:
                    print("Image loading error in model_evaluate...",img_path)
                
                # resize image
                img = np.array(img)
                img = img.reshape((1, img_width, img_height, img_channel))  
  
            else:                                               # take from test_image
                print("Take image from test_image...")
                
                # take image
                img = cv2.imread(test_image[index_image_visualized])            # take current iamge
                if img is not None:                     # check image taken
                    #check if the image is in the correct shape for the CNN (shape specified in the global variables)
                    if img.shape != (img_width, img_height, img_channel):       
                        dim = (img_height ,img_width)
                        img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                        img = img.astype('float32') / 255                           # normalization
                    else:
                        img = img.astype('float32') / 255                           # normalization
                else:
                    print("Image loading error in model_evaluate...",img_path)
                
                # resize image
                img = np.array(img)
                img = img.reshape((1, img_width, img_height, img_channel))  

            # predict
            predictions = network.predict(img)                              # get the output for each sample
            classify_text.set(': '+str(classes[np.argmax(predictions)]))    # update GUI
        
        elif label_ext_image_text.get() != '':                  # image from external test set
            # take image
            img = cv2.imread(test_image_ext[index_image_visualized])            # take current iamge
            if img is not None:                     # check image taken
                #check if the image is in the correct shape for the CNN (shape specified in the global variables)
                if img.shape != (img_width, img_height, img_channel):       
                    dim = (img_height ,img_width)
                    img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                    img = img.astype('float32') / 255                           # normalization
                else:
                    img = img.astype('float32') / 255                           # normalization
            else:
                 print("Image loading error in model_evaluate...",img_path)
                
            # resize image
            img = np.array(img)
            img = img.reshape((1, img_width, img_height, img_channel))  
            # predict
            predictions = network.predict(img)                              # get the output for each sample
            classify_ext_text.set(': '+str(classes[np.argmax(predictions)]))# update GUI
    else:
        error_text.set(er_predict_text)                         # update text error
# ------------------------------------ end: methods for CNN model ------------------------------------

# ------------------------------------ start: utility method ------------------------------------
# method that verify if the past model has the same parameters of the actual model call
def check_past_model(type_model):
    global past_param_model

    if past_param_model["type_model"] == type_model:          # check type of the model
        if past_param_model["epochs"] == epochs:                  # save type of the model
            if past_param_model["batch_size"] == batch_size:          # save type of the model
                if past_param_model["early_patience"] == early_patience:   # save type of the model
                    return True                                            # they are the same
    return False                                    # they aren't the same

def plot_accuracy_and_loss(train_hist, val_hist, test_hist):
    runs = range(1, len(train_hist['accuracy']) + 1)
    
    # -- plot accuracy --
    plt.figure(figsize=(10, 6))
    plt.plot(runs, train_hist['accuracy'], label='Train Accuracy')
    plt.plot(runs, val_hist['accuracy'], label='Validation Accuracy')
    plt.plot(runs, test_hist['accuracy'], label='Test Accuracy')
    
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per run')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # -- plot loss --
    plt.figure(figsize=(10, 6))
    plt.plot(runs, train_hist['loss'], label='Train Loss')
    plt.plot(runs, val_hist['loss'], label='Validation Loss')
    plt.plot(runs, test_hist['loss'], label='Test Loss')
    
    plt.xlabel('Run')
    plt.ylabel('Loss')
    plt.title('Loss per run')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_average_metrics(train_hist, val_hist, test_hist):
    def avg(lst): return round(np.mean(lst), 4)
    
    print("\nMean:")
    print(f"Train     → Accuracy: {avg(train_hist['accuracy'])}, Loss: {avg(train_hist['loss'])}")
    print(f"Validation→ Accuracy: {avg(val_hist['accuracy'])}, Loss: {avg(val_hist['loss'])}")
    print(f"Test      → Accuracy: {avg(test_hist['accuracy'])}, Loss: {avg(test_hist['loss'])}")


# method to plot accuracy and loss. arc is a dictionary with the results, 'mode' if is '0': there are fit results, if is '1': there are evaluation results
def plot_fit_result(arc,mode):
    result_dict = {}                    # dict that will contain the results to plot with the correct label/title
    # check what results there are
    if mode == 0:                       # method called with fit results
        result_dict["loss (training set)"] = arc["loss"]                    # take loss values (training set)
        result_dict["accuracy (taining set)"] = arc["accuracy"]             # take accuracy values (training set)
        if arc.get("val_loss") is not None:                         # check if there are result of validation set
            result_dict["loss (validation set)"] = arc["val_loss"]          # take loss values (validation set)
            result_dict["accuracy (validation set)"] = arc["val_accuracy"]  # take accuracy values (validation set)
    elif mode == 1:                     # method called with evaluate results
        result_dict["loss (test set)"] = arc["loss"]                        # take loss values (test set)
        result_dict["accuracy (test set)"] = arc["accuracy"]                # take accuracy values (test set)
    # plot the results
    for k,v in result_dict.items():
        #print("chiave: ", k," value: ",v)
        plot(k,v)

# method to display a plot. 'title' is the tile of the plot, 'value_list' is a list of value to draw in the plot
def plot(title,value_list):
    fig = plt.figure()
    fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))    # force the label of  number of epochs to be integer
    plt.plot(value_list,'o-b')
    plt.title(str(title))               # plot title
    plt.xlabel("# Epochs")              # x axis title
    plt.ylabel("Value")                 # y axis title
    plt.show()
    
# method for create and plot the confusion metrix of the model trained
def confusion_matrix():
    global test_image, test_label, network, classes # global variables references
    data_test = []                                  # set of the images of the test set
    
    # create the confusion matrix, rows indicate the real class and columns indicate the predicted class 
    conf_matrix = np.zeros((len(classes),len(classes)))     # at begin values are 0
    
    # take all the images from the test set
    for img_path in test_image:
        img = cv2.imread(img_path)              # take current iamge
        if img is not None:                     # check image taken
        #check if the image is in the correct shape for the CNN (shape specified in the global variables)
            if img.shape != (img_width, img_height, img_channel):       
                dim = (img_height ,img_width)
                img = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)   # resize the image
                img = img.astype('float32') / 255                           # normalization
                data_test.append(img)                                       # add new element
            else:
                img = img.astype('float32') / 255                           # normalization
                data_test.append(img)                                       # add new element
        else:
            print("Image loading error in model_evaluate...",img_path)
                    
    # resize the set of the image
    data_test = np.array(data_test)
    data_test = data_test.reshape((len(data_test), img_width, img_height, img_channel))     
    
    predictions = network.predict(data_test)                # get the output for each sample of the test set
    # slide the prediction result and go to create the confusion matrix
    for i in range(len(data_test)):
        # test_label[i] indicate the real value of the label associated at the test_image[i] (or data_test[i]) -> is real class (row)
        # predictions[i] indicate the class value predicted by the model for the test_image[i] (or data_test[i]) -> is predicted class (column)
        # the values are in categorical format, translate in int
        conf_matrix[np.argmax(test_label[i])][np.argmax(predictions[i])] += 1                               # update value
        
    # do percentages of confusion matrix
    conf_matrix_perc = [[None for c in range(conf_matrix.shape[1])] for r in range(conf_matrix.shape[0])]   # define matrix
    
    for i in range(conf_matrix.shape[0]):                   # rows
        for j in range(conf_matrix.shape[1]):               # columns
            conf_matrix_perc[i][j] = " (" + str( round( (conf_matrix[i][j]/len(data_test))*100 ,2) ) + "%)" # calculate percentage value
    
    # plot the confusion matrix
    rows = classes                                          # contain the label of the classes showed in the rowvalues of rows          
    columns = classes                                       # contain the label of the classes showed in the rowvalues of columns   

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    
    for i in range(len(rows)):                              # rows
        for j in range(len(columns)):                       # columns
            # give the value in the confusion matrix
            ax.text(x=j, y=i, s=str(str(conf_matrix[i][j])+conf_matrix_perc[i][j]),
                           ha="center", va="center", size='x-large')
            
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Real', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()                                              # shows confusion matrix
    
# to calculate confusion matrix using dataset.from.generator 
def compute_confusion_matrix(model, dataset, steps, classes):
    # Previsione batch-wise usando il dataset
    predictions = model.predict(dataset, steps=steps)

    # Estrazione dei dati reali (labels) dal dataset (solo per i batch usati)
    true_labels = []
    for batch_idx, (_, y) in enumerate(dataset):
        true_labels.append(y.numpy())
        if batch_idx + 1 >= steps:
            break
    true_labels = np.concatenate(true_labels, axis=0)

    # Calcolo confusion matrix
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(predictions)):
        true_idx = np.argmax(true_labels[i])
        pred_idx = np.argmax(predictions[i])
        conf_matrix[true_idx][pred_idx] += 1

    # Calcolo percentuali
    conf_matrix_perc = [[
        f" ({(conf_matrix[i][j] / np.sum(conf_matrix)) * 100:.2f}%)"
        for j in range(conf_matrix.shape[1])
    ] for i in range(conf_matrix.shape[0])]

    # Visualizzazione confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{conf_matrix[i][j]}{conf_matrix_perc[i][j]}",
                    va='center', ha='center', fontsize=12)

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()
    
# method to check GPU device avaible and setting
def GPU_check():
    print("-------------------- TENSORFLOW VERSION --------------------")
    print(tf.__version__)
    print("-------------------- AVAILABLE HW DEVICES --------------------")
    print("List of devices:")
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)    
    print("Used VRAM (GPU memory): ",print("",tf.config.experimental.get_memory_info('GPU:0')['current'] / 10**9))
    print("------------------------------------------------------------")

# utility function to print the VRAM used at the moment
def gpu_memory_usage():
    print("-------------------- VRAM USAGE --------------------")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for i, gpu in enumerate(gpus):
                info = tf.config.experimental.get_memory_info(f"GPU:{i}")
                used = info['current'] / (1024**2)  # in MB
                print(f"GPU {i} VRAM used: {used:.2f} MB")
        except Exception as e:
            print("Errore nel monitorare la GPU:", e)
    print("------------------------------------------------------------")

# funzione di utilità per visualizzare un tempo dato in secondi in minuti, secondi
def converti_secondi(sec):
    ore = sec // 3600
    minuti = (sec // 60) % 60
    secondi = sec % 60
    return f"{ore:.0f}h {minuti:.0f}m {secondi:.2f}s"
# ------------------------------------ end: utility method ------------------------------------

# ------------------------------------ main ------------------------------------        
if __name__ == "__main__":
    window.title("GenCNNClassifier")                            # title of the GUI window
    window.geometry(str(window_width)+'x'+str(window_height))   # size of the windows
    window.resizable(False, False)                              # window can't be resized
    
    GPU_check()                     # check if the program can be read the GPU (the library for AI of Nvidia is setting properly)
    #set_background_image()         # set background image
    current_view_to_visualise()     # method for visualize the correct GUI
    
    # handle the window closing by the user
    #window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()
    
# ------------------------------------ Notes ------------------------------------
# -- Note 0 --
# in "import_image_from_ds" method I chose an approch more flexible than necessary. Array classes will contain the labels of the classes,
# in this way the classes number and names are not fixed apriori but it's calculated at real time.

# -- Note 1 --