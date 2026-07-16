# -*- coding: utf-8 -*-
"""
Created on 2026 (PyTorch port)
@author: Alessandro Diana

explanation:
    PyTorch version of genCNNClassifier. A Tkinter GUI to train and test CNNs
    and to visually verify the classification of test-set images. It keeps the
    same feature set as the TensorFlow version (dataset load/analyse, model
    build/fit, evaluation, single-image prediction, save/load, confusion matrix,
    plots) but the training engine is fully PyTorch based.

Main differences vs the TensorFlow version:
    - Images are handled channels-first (N, C, H, W).
    - Batching uses torch.utils.data.Dataset + DataLoader instead of a
      tf.data generator + steps_per_epoch. This natively avoids loading the
      whole dataset in memory (images are read lazily inside __getitem__).
    - Labels are integer class indices (not one-hot); loss is nn.CrossEntropyLoss.
    - Early stopping / best-weights checkpointing are implemented in a manual
      training loop.
    - GoogLeNet returns three heads during training; the total loss is
      main + 0.3*aux1 + 0.3*aux2.
"""

# ---- general ----
import os
import numpy as np
import random
import time
import math
import shutil
import copy

# ---- torch ----
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---- plot ----
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- image ----
import cv2
import PIL
from PIL import ImageTk
from PIL import Image

# ---- GUI ----
from tkinter import *
from tkinter import ttk

# ---- thread ----
from threading import Thread

# ---- my network classes ----
from net_classes import AlexNet_class as ANet
from net_classes import GoogLeNet_class as GLNet
from net_classes import IfritNet_class as IfritNet

# device used for training / inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------ start: global var ------------------------------------
# ---- GUI variables ----
window = Tk()
toggle_duplicate_ds = BooleanVar(value=False)   # create a resized copy of the dataset
toggle_grey_scale = BooleanVar(value=False)     # create a greyscale copy of the dataset
window_width = 1800
window_height = 750
# explain frame
ex_f_padx = 10;  ex_f_pady = 5
ex_frame_width = window_width - 2 * ex_f_padx;  ex_frame_height = 45
# top frame
t_f_padx = 10;  t_f_pady = 5
top_frame_width = window_width - 2 * t_f_padx;  top_frame_height = 180
# image frame
im_f_padx = 10;  im_f_pady = 5
im_frame_width = window_width - 2 * im_f_padx;  im_frame_height = 250
# bottom frame
b_f_padx = 10;  b_f_pady = 5
b_frame_width = window_width - 2 * im_f_padx;  b_frame_height = 170
# error frame
er_f_padx = 10;  er_f_pady = 5
er_frame_width = window_width - 2 * im_f_padx;  er_frame_height = 45

# ---- errors variables ----
error_text = StringVar(); error_text.set('')
er_load_model_text = "Please insert a model name in the field before load CNN model."
er_load_model_unknown_text = "There isn't a CNN model with the specified name."
er_save_model_text = "Please insert a model name in the field before save CNN model."
er_no_ds_text = "Please load the dataset and retry."
er_no_ext_ds_text = "Please load the external test dataset and retry."
er_train_without_ds_text = "Before train the model you must load image dataset."
er_eval_without_model_text = "Before evaluate the model you must make and fit or load a model."
er_no_model_specified_text = "Please chose a CNN model or load one before fit CNN model."
er_predict_text = "Before predict you must train model and load an image."
er_format_epoch_text = "Error format in the Number of epochs input, you must insert a positive number, please retry."
er_format_batch_size_text = "Error format in the Number of batch_size input, you must insert a positive number, please retry."
er_format_image_size_text = "Error format in the Numbers for the image size, you must insert a positive number, please retry."
er_format_early_text = "Error format in the Number of early patience input, you must insert a positive number, please retry."
er_down_ds_text = "Dataset downloading already started, please wait."

# ---- status variables ----
model_trained = False
image_to_visualize = None
index_image_visualized = -1
ds_downloading = False
ext_ds_downloading = False
status_DS_text = StringVar();          status_DS_text.set('Image DataSet: missing')
status_ext_test_DS_text = StringVar(); status_ext_test_DS_text.set('External test DS: missing')
CNN_menu_text = StringVar();           CNN_menu_text.set('None')
status_model_text = StringVar();       status_model_text.set('CNN model: empty')

# ---- label text variables ----
classify_text = StringVar();     classify_text.set('')
classify_ext_text = StringVar(); classify_ext_text.set('')
label_image_text = StringVar();  label_image_text.set('')
label_ext_image_text = StringVar(); label_ext_image_text.set('')

# ---- path variables ----
path_dir_ds = os.path.join("Dataset", "polmonite")       # training dataset folder
path_dir_test_ds = os.path.join("Dataset", "pkm_test_ds") # external test dataset folder
path_dir_model = "Model"                                  # saved models folder
path_check_point_model = os.path.join(path_dir_model, "train_ckpt")  # checkpoints folder
res_copy_ds_name = "res_ds_copy"

# ---- model variables ----
network = None                      # the nn.Module currently loaded/trained
truncate_set = False                # kept for parity (DataLoader handles the last batch)
past_param_model = {"type_model": None, "epochs": None, "batch_size": None, "early_patience": None}
batch_size = 32
def_img_height = 224; def_img_width = 224; def_img_channel = 3
img_height = def_img_height; img_width = def_img_width; img_channel = def_img_channel
epochs = 100
early_patience = 10

# ---- dataset variables ----
classes = []
total_image_ds = []
total_labels_ds = []
train_image = []; train_label = []
val_img = [];     val_label = []
test_image = [];  test_label = []
test_image_ext = []; test_label_ext = []
test_set_split = 0.2
val_set_split = 0.1

# ---- dataset information variables ----
del_corrupt_img = False
ds_analysing = False
RGB_mean_dict = {}
# ------------------------------------ end: global var ------------------------------------


# ------------------------------------ start: methods for GUI ------------------------------------
def cleanGUI():
    for l in window.grid_slaves():
        l.destroy()


def on_closing():
    window.destroy()


def current_view_to_visualise():
    cleanGUI()
    explainText = "Welcome to genCNNClassifier (PyTorch).\nA general program to train and test CNNs."
    CNN_model_text = ['None', 'AlexNet', 'GoogleNet', 'Ifrit_1', 'Ifrit_2', 'Ifrit_3', 'Ifrit_4']

    # ---- explain frame ----
    explain_frame = Frame(window, width=ex_frame_width, height=ex_frame_height, bg='grey')
    explain_frame.grid(row=0, column=0, padx=ex_f_padx, pady=ex_f_pady, sticky="nsew")
    explain_frame.grid_propagate(False)
    Label(explain_frame, text=explainText).grid(row=0, column=0, sticky="W", padx=window_width / 2, pady=5)

    # ---- top frame ----
    top_frame = Frame(window, width=top_frame_width, height=top_frame_height, bg='grey')
    top_frame.grid(row=1, column=0, padx=t_f_padx, pady=t_f_pady, sticky="nsew")
    top_frame.grid_propagate(False)

    # row 0
    Label(top_frame, text="DS name (in 'dataset' folder): ").grid(row=0, column=0, sticky="W", padx=5, pady=10)
    path_ds_input = Entry(top_frame, width=15); path_ds_input.grid(row=0, column=1, sticky="WE", padx=5)
    Label(top_frame, text="Input image size: ").grid(row=0, column=2, sticky="W", padx=5, pady=10)
    image_width_input = Entry(top_frame, width=15);   image_width_input.grid(row=0, column=3, sticky="WE", padx=5)
    image_height_input = Entry(top_frame, width=15);  image_height_input.grid(row=0, column=4, sticky="WE", padx=5)
    image_channel_input = Entry(top_frame, width=15); image_channel_input.grid(row=0, column=5, sticky="WE", padx=5)
    Label(top_frame, text="Default size: (" + str(def_img_width) + "," + str(def_img_height) + "," + str(def_img_channel) + ")").grid(row=0, column=6, sticky="W", padx=5, pady=10)
    Button(top_frame, text="Analyse DS", command=lambda: btn_analyse_ds(path_ds_input.get(), image_width_input.get(), image_height_input.get(), image_channel_input.get())).grid(row=0, column=8, sticky="W", padx=5, pady=10)
    Checkbutton(top_frame, text="Resized DS copy", variable=toggle_duplicate_ds).grid(row=0, column=9, sticky="W", padx=5, pady=10)
    Checkbutton(top_frame, text="Greyscale DS copy", variable=toggle_grey_scale).grid(row=0, column=10, sticky="W", padx=5, pady=10)
    Button(top_frame, text="Load image DS", command=lambda: btn_load_ds_method(image_width_input.get(), image_height_input.get(), image_channel_input.get())).grid(row=0, column=11, sticky="W", padx=5, pady=10)

    # row 1
    Label(top_frame, text="CNN model name: ").grid(row=1, column=6, sticky="W", padx=5, pady=10)
    name_model_input = Entry(top_frame, width=15); name_model_input.grid(row=1, column=7, sticky="WE", padx=5)
    Button(top_frame, text="Load CNN model", command=lambda: load_saved_model(name_model_input.get())).grid(row=1, column=8, sticky="W", padx=5, pady=10)
    Button(top_frame, text="Save CNN model", command=lambda: save_model(name_model_input.get())).grid(row=1, column=9, sticky="W", padx=5, pady=10)
    Button(top_frame, text="Fit CNN model", command=lambda: make_fit_model(CNN_menu_text.get(), number_epochs_input.get(), batch_size_input.get(), early_stopping_input.get(), num_train_input.get())).grid(row=1, column=12, sticky="W", padx=5, pady=10)

    # row 2
    Label(top_frame, text="Select CNN model:").grid(row=2, column=0, sticky="W", padx=5, pady=10)
    OptionMenu(top_frame, CNN_menu_text, *CNN_model_text).grid(row=2, column=1, padx=10)
    Label(top_frame, text="Number of epoch:").grid(row=2, column=2, sticky="W", padx=5, pady=10)
    number_epochs_input = Entry(top_frame, width=10); number_epochs_input.grid(row=2, column=3, sticky="WE", padx=5)
    Label(top_frame, text="Batch size:").grid(row=2, column=4, sticky="W", padx=5, pady=10)
    batch_size_input = Entry(top_frame, width=10); batch_size_input.grid(row=2, column=5, sticky="WE", padx=5)
    Label(top_frame, text="Early patience:").grid(row=2, column=6, sticky="W", padx=5, pady=10)
    early_stopping_input = Entry(top_frame, width=10); early_stopping_input.grid(row=2, column=7, sticky="WE", padx=5)
    Label(top_frame, text="Number of train to do:").grid(row=2, column=8, sticky="W", padx=5, pady=10)
    num_train_input = Entry(top_frame, width=10); num_train_input.grid(row=2, column=9, sticky="WE", padx=5)

    # row 3 (status)
    Label(top_frame, textvariable=status_model_text).grid(row=3, column=2, sticky="W", padx=5, pady=10)
    Label(top_frame, textvariable=status_DS_text).grid(row=3, column=3, sticky="W", padx=5, pady=10)

    # ---- image frame ----
    image_frame = Frame(window, width=im_frame_width, height=im_frame_height, bg='grey')
    image_frame.grid(row=2, column=0, padx=im_f_padx, pady=im_f_pady, sticky="nsew")
    image_frame.grid_propagate(False)
    if image_to_visualize is not None:
        Label(image_frame, image=image_to_visualize).grid(row=0, column=1, sticky="W", padx=(window_width / 2 - img_width / 2), pady=5)

    # ---- bottom frame ----
    bottom_frame = Frame(window, width=b_frame_width, height=b_frame_height, bg='grey')
    bottom_frame.grid(row=3, column=0, padx=b_f_padx, pady=b_f_pady, sticky="nsew")
    bottom_frame.grid_propagate(False)
    # row 0
    Button(bottom_frame, text="Take image", command=btn_load_image).grid(row=0, column=0, sticky="W", padx=10, pady=10)
    Label(bottom_frame, text="Label: ", width=9).grid(row=0, column=1, sticky="W", padx=10, pady=10)
    Label(bottom_frame, textvariable=label_image_text, width=9).grid(row=0, column=2, sticky="W", padx=10, pady=10)
    Button(bottom_frame, text="Classify:", command=predict, width=11).grid(row=0, column=3)
    Label(bottom_frame, textvariable=classify_text, width=9).grid(row=0, column=4, sticky="W", padx=10, pady=5)
    # row 1
    Button(bottom_frame, text="Load extern test DS", command=lambda: btn_load_ext_ds_method(image_width_input.get(), image_height_input.get(), image_channel_input.get())).grid(row=1, column=0, sticky="W", padx=10, pady=10)
    Label(bottom_frame, textvariable=status_ext_test_DS_text).grid(row=1, column=5, sticky="W", padx=10, pady=10)
    # row 2
    Button(bottom_frame, text="Take extern test image", command=btn_load_ext_image).grid(row=2, column=0, sticky="W", padx=10, pady=10)
    Label(bottom_frame, text="Label: ", width=9).grid(row=2, column=1, sticky="W", padx=10, pady=10)
    Label(bottom_frame, textvariable=label_ext_image_text, width=9).grid(row=2, column=2, sticky="W", padx=10, pady=10)
    Button(bottom_frame, text="Classify:", command=predict, width=11).grid(row=2, column=3)
    Label(bottom_frame, textvariable=classify_ext_text, width=9).grid(row=2, column=4, sticky="W", padx=10, pady=5)
    # row 3
    Button(bottom_frame, text="Evaluate CNN", command=lambda: model_evaluate("test")).grid(row=3, column=1)
    Button(bottom_frame, text="Evaluate CNN (extern)", command=lambda: model_evaluate("extern")).grid(row=3, column=3)

    # ---- error frame ----
    error_frame = Frame(window, width=er_frame_width, height=er_frame_height, bg='grey')
    error_frame.grid(row=4, column=0, padx=er_f_padx, pady=er_f_pady, sticky="nsew")
    error_frame.grid_propagate(False)
    Label(error_frame, textvariable=error_text, bg='grey').grid(row=0, column=0, padx=10, pady=10)


def _show_image_in_gui(img_path):
    """Load, resize and display an image path in the GUI image frame."""
    global image_to_visualize
    img = cv2.imread(img_path)
    if img is None:
        return
    dim = ((im_frame_height - 2 * im_f_pady), (im_frame_height - 2 * im_f_pady))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    blue, green, red = cv2.split(img)
    img = cv2.merge((red, green, blue))
    im = PIL.Image.fromarray(np.uint8(img), 'RGB')
    image_to_visualize = ImageTk.PhotoImage(image=im)
    current_view_to_visualise()


def btn_load_image():
    global index_image_visualized
    error_text.set('')
    if len(test_image) == 0 or len(test_label) == 0:
        if len(total_image_ds) != 0:
            index = random.randint(0, len(total_image_ds) - 1)
            index_image_visualized = index
            label_image_text.set(str(classes[int(total_labels_ds[index])]))
            label_ext_image_text.set('')
            _show_image_in_gui(total_image_ds[index])
        else:
            error_text.set(er_no_ds_text)
    else:
        index = random.randint(0, len(test_image) - 1)
        index_image_visualized = index
        label_image_text.set(str(classes[int(np.argmax(test_label[index]))]))
        label_ext_image_text.set('')
        _show_image_in_gui(test_image[index])


def btn_load_ext_image():
    global index_image_visualized
    error_text.set('')
    if len(test_image_ext) == 0 or len(test_label_ext) == 0:
        error_text.set(er_no_ext_ds_text)
    else:
        index = random.randint(0, len(test_image_ext) - 1)
        index_image_visualized = index
        label_ext_image_text.set(str(classes[int(test_label_ext[index])]))
        label_image_text.set('')
        _show_image_in_gui(test_image_ext[index])
# ------------------------------------ end: methods for GUI ------------------------------------


# ------------------------------------ start: methods for DS ------------------------------------
def btn_load_ds_method(im_width, im_height, im_channel):
    global ds_downloading
    error_text.set('')
    if not (ds_downloading or ds_analysing):
        ds_downloading = True
        Thread(target=import_image_from_ds, args=(path_dir_ds, im_width, im_height, im_channel)).start()
    else:
        error_text.set(er_down_ds_text)


def import_image_from_ds(path_ds, im_width, im_height, im_channel):
    """Import the whole dataset. Only image paths and integer labels are stored;
    the pixels are read lazily during training (see ImageFolderDataset)."""
    global total_image_ds, total_labels_ds, ds_downloading, classes, img_channel
    list_dir_ds = os.listdir(path_ds)
    make_ds_copy = toggle_duplicate_ds.get()

    if len(total_image_ds) != 0:
        classes = []; total_image_ds = []; total_labels_ds = []

    if not check_image_size_param(im_width, im_height, im_channel):
        ds_downloading = False
        return

    if toggle_grey_scale.get():
        img_channel = 1
        toggle_duplicate_ds.set(True)
        make_ds_copy = True

    path_copy = None
    if make_ds_copy:
        parent_path = os.path.dirname(path_ds)
        path_copy = os.path.join(parent_path, res_copy_ds_name)
        if os.path.exists(path_copy):
            shutil.rmtree(path_copy)
        os.makedirs(path_copy)

    status_DS_text.set('Image DataSet: loading')
    print("-------------------- READING DS --------------------")
    print("Read the DS: ", path_ds)

    for folder in list_dir_ds:
        print("Read the folder: ", folder)
        classes.append(str(folder))
        index = classes.index(str(folder))
        p = os.path.join(path_ds, folder)
        if make_ds_copy:
            p_copy = os.path.join(path_copy, folder)
            os.makedirs(p_copy, exist_ok=True)
        for filename in os.listdir(p):
            src = os.path.join(p, filename)
            if make_ds_copy:
                if toggle_grey_scale.get():
                    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(src)
                if img is None:
                    print("Image loading error...", filename); continue
                if img.shape[:2] != (img_height, img_width):
                    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
                out_path = os.path.join(p_copy, filename)
                cv2.imwrite(out_path, img)
                total_image_ds.append(out_path)
                total_labels_ds.append(index)
            else:
                img = cv2.imread(src)
                if img is not None:
                    total_image_ds.append(src)
                    total_labels_ds.append(index)
                else:
                    print("Image loading error...", filename)

    total_image_ds = np.array(total_image_ds)
    total_labels_ds = np.array(total_labels_ds)
    print("Num of classes: ", len(classes))
    print("total_image_ds", len(total_image_ds), total_image_ds.shape)
    print("total_labels_ds", len(total_labels_ds), total_labels_ds.shape)
    print("------------------------------------------------------------")
    status_DS_text.set('Image DataSet: downloaded')
    error_text.set('')
    ds_downloading = False


def btn_load_ext_ds_method(im_width, im_height, im_channel):
    global ext_ds_downloading
    error_text.set('')
    if not (ext_ds_downloading or ds_analysing):
        ext_ds_downloading = True
        Thread(target=import_image_from_ext_test_ds, args=(path_dir_test_ds, im_width, im_height, im_channel)).start()
    else:
        error_text.set(er_down_ds_text)


def import_image_from_ext_test_ds(path_ds, im_width, im_height, im_channel):
    global test_image_ext, test_label_ext, ext_ds_downloading
    image_ds = []; labels_ds = []
    list_dir_ds = os.listdir(path_ds)
    error_text.set('')
    if len(total_image_ds) == 0:
        error_text.set(er_no_ds_text)
        ext_ds_downloading = False
        return
    if not check_image_size_param(im_width, im_height, im_channel):
        ext_ds_downloading = False
        return
    status_ext_test_DS_text.set('External test DS: loading')
    for folder in list_dir_ds:
        print("Read the folder: ", folder)
        index = classes.index(str(folder))
        p = os.path.join(path_ds, folder)
        for filename in os.listdir(p):
            src = os.path.join(p, filename)
            if cv2.imread(src) is not None:
                image_ds.append(src); labels_ds.append(index)
            else:
                print("Image loading error...", filename)
    test_image_ext = np.array(image_ds)
    test_label_ext = np.array(labels_ds)
    print("test_image_ext", len(test_image_ext), test_image_ext.shape)
    status_ext_test_DS_text.set('External test DS: downloaded')
    error_text.set('')
    ext_ds_downloading = False


def btn_analyse_ds(folder_name, im_width, im_height, im_channel):
    global ds_analysing
    error_text.set('')
    if not (ds_downloading or ds_analysing):
        ds_analysing = True
        t = Thread(target=analyse_ds, args=(folder_name, im_width, im_height, im_channel))
        t.start()
        check_thread(t, window)
    else:
        error_text.set(er_down_ds_text)


def check_thread(thread, root):
    if thread.is_alive():
        root.after(100, lambda: check_thread(thread, root))
    else:
        show_plot_analysing()
        status_DS_text.set('Image DataSet: analysed')


def show_plot_analysing():
    for k, v in RGB_mean_dict.items():
        color = ["Blue", "Green", "Red"]
        x_pos = np.arange(len(color))
        plt.bar(x_pos, v, align='center')
        plt.xticks(x_pos, color)
        plt.ylabel('Value'); plt.xlabel('Color'); plt.title(k)
        plt.show()


def analyse_ds(folder_name, im_width, im_height, im_channel):
    global ds_analysing, RGB_mean_dict
    img_number = 0
    classes_count = {}
    format_dict = {}
    shape_dict = {}
    top_shape_images = 10
    corrupt_images = []
    RGB_mean_dict = {}

    if is_valid_folder_name(folder_name) and os.path.isdir(os.path.join("Dataset", folder_name)):
        new_path_ds = os.path.join("Dataset", folder_name)
    else:
        new_path_ds = path_dir_ds
    list_dir_ds = os.listdir(new_path_ds)

    if not check_image_size_param(im_width, im_height, im_channel):
        ds_analysing = False
        return

    status_DS_text.set('Image DataSet: analysing')
    print("-------------------- ANALYSING DS --------------------")
    print(" -- Analyse the DS: ", new_path_ds, " --")

    for folder in list_dir_ds:
        print("Analyse the folder: ", folder)
        if classes_count.get(str(folder)) is None:
            classes_count[str(folder)] = 0
        p = os.path.join(new_path_ds, folder)
        for filename in os.listdir(p):
            img_number += 1
            classes_count[str(folder)] += 1
            format_name = str(filename).split('.')[-1]
            format_dict[format_name] = format_dict.get(format_name, 0) + 1
            img = cv2.imread(os.path.join(p, filename))
            if img is not None:
                shape_dict[str(img.shape)] = shape_dict.get(str(img.shape), 0) + 1
                red_mean = img[:, :, 2].mean()
                green_mean = img[:, :, 1].mean()
                blue_mean = img[:, :, 0].mean()
                if RGB_mean_dict.get(str(folder)) is None:
                    RGB_mean_dict[str(folder)] = [blue_mean, green_mean, red_mean]
                else:
                    t = RGB_mean_dict[str(folder)]
                    RGB_mean_dict[str(folder)] = [t[0] + blue_mean, t[1] + green_mean, t[2] + red_mean]
            else:
                corrupt_images.append(str(filename))
                if del_corrupt_img:
                    os.remove(os.path.join(p, filename))

    print("The dataset characteristics are:")
    print("- The dataset has ", img_number, " images.")
    print("- The number of classes are ", len(classes_count.keys()), ". ", list(classes_count.keys()))
    print("List of the classes and number of related images for each one:")
    for k, v in dict(sorted(classes_count.items(), key=lambda x: x[1], reverse=True)).items():
        print('- ', k, ': ', v)
    if len(corrupt_images) != 0:
        print("There are ", len(corrupt_images), " corrupted images.\n", corrupt_images)
    print("The number of different images shapes are ", len(shape_dict.keys()))
    count = 0
    for k, v in dict(sorted(shape_dict.items(), key=lambda x: x[1], reverse=True)).items():
        if count >= top_shape_images:
            break
        print('- ', k, ': ', v); count += 1
    print("The number of different images format are ", len(format_dict.keys()))
    count = 0
    for k, v in dict(sorted(format_dict.items(), key=lambda x: x[1], reverse=True)).items():
        if count >= top_shape_images:
            break
        print('- ', k, ': ', v); count += 1
    ds_analysing = False
    print("------------------------------------------------------------")


def make_set_ds():
    """Split total_image_ds/total_labels_ds into train/val/test index arrays.
    Labels stay as integer class indices (no one-hot: CrossEntropyLoss)."""
    global test_image, test_label, train_image, train_label, val_img, val_label
    seed = random.randint(1, 42)
    image_ds = np.array(total_image_ds)
    labels_ds = np.array(total_labels_ds).astype(np.int64)

    train_img_temp, test_image, train_label_temp, test_label = train_test_split(
        image_ds, labels_ds, test_size=test_set_split, random_state=seed, shuffle=True)
    train_image, val_img, train_label, val_label = train_test_split(
        train_img_temp, train_label_temp, test_size=test_set_split, random_state=seed, shuffle=True)

    print("-------------------- SETS --------------------")
    print("train_image len:", len(train_image))
    print("val_img len:", len(val_img))
    print("test_image len:", len(test_image))
    print("------------------------------------------------------------")
# ------------------------------------ end: methods for DS ------------------------------------


# ------------------ start: PyTorch Dataset ------------------
class ImageFolderDataset(Dataset):
    """Lazy dataset: reads and preprocesses each image on demand, so the whole
    dataset never sits in memory at once (equivalent purpose to the old
    tf.data generator). Returns (tensor[C,H,W] float32, label int64)."""

    def __init__(self, image_paths, labels, use_greyscale):
        self.image_paths = image_paths
        self.labels = np.array(labels).astype(np.int64)
        self.use_greyscale = use_greyscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if self.use_greyscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
        if img is None:
            # return a zero tensor to avoid crashing the batch
            c = 1 if self.use_greyscale else img_channel
            img = np.zeros((img_height, img_width, c), dtype=np.uint8)
        if img.shape[:2] != (img_height, img_width):
            img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
        img = img.astype('float32') / 255.0
        if img.ndim == 2:                       # greyscale -> add channel axis
            img = img[:, :, None]
        img = np.transpose(img, (2, 0, 1))      # HWC -> CHW
        return torch.from_numpy(img), int(self.labels[idx])
# ------------------ end: PyTorch Dataset ------------------


# ------------------------------------ start: methods for CNN model ------------------------------------
def check_image_size_param(height, width, channel):
    global img_height, img_width, img_channel
    for value, setter in [(height, 'h'), (width, 'w'), (channel, 'c')]:
        if value:
            if value.isnumeric():
                try:
                    iv = int(value)
                except Exception:
                    error_text.set(er_format_image_size_text); return False
                if setter == 'h': img_height = iv
                elif setter == 'w': img_width = iv
                else: img_channel = iv
            else:
                error_text.set(er_format_image_size_text); return False
    return True


def is_valid_folder_name(name):
    if not name.strip():
        return False
    invalid_chars = '<>:"/\\|?*'
    if any(char in name for char in invalid_chars):
        return False
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'COM{i}' for i in range(1, 10)} | {f'LPT{i}' for i in range(1, 10)}
    return name.upper() not in reserved


def check_fit_param(number_epoch, num_batch_size, num_early_patience):
    global epochs, batch_size, early_patience
    if number_epoch:
        if number_epoch.isnumeric():
            epochs = int(number_epoch)
        else:
            error_text.set(er_format_epoch_text); return False
    if num_batch_size:
        if num_batch_size.isnumeric():
            batch_size = int(num_batch_size)
        else:
            error_text.set(er_format_batch_size_text); return False
    if num_early_patience:
        if num_early_patience.isnumeric():
            early_patience = int(num_early_patience)
        else:
            error_text.set(er_format_early_text); return False
    return True


def validate_num_test(value):
    try:
        num = int(value)
        return num if num > 0 else 1
    except (ValueError, TypeError):
        return 1


def _build_network(chosen_model):
    """Instantiate the chosen architecture and move it to the training device."""
    if chosen_model == "AlexNet":
        m = ANet.AlexNet(len(classes), img_width, img_height, img_channel); m.make_model()
    elif chosen_model == "GoogleNet":
        m = GLNet.GoogLeNet(len(classes), img_width, img_height, img_channel); m.make_model()
    elif chosen_model == "Ifrit_1":
        m = IfritNet.IfritNet(len(classes), img_width, img_height, img_channel); m.make_model(1)
    elif chosen_model == "Ifrit_2":
        m = IfritNet.IfritNet(len(classes), img_width, img_height, img_channel); m.make_model(2)
    elif chosen_model == "Ifrit_3":
        m = IfritNet.IfritNet(len(classes), img_width, img_height, img_channel); m.make_model(3)
    elif chosen_model == "Ifrit_4":
        m = IfritNet.IfritNet(len(classes), img_width, img_height, img_channel); m.make_model(4)
    else:
        return None
    return m.to(DEVICE)


def _optimizer_for(chosen_model, model):
    # AlexNet used rmsprop in the TF version; everything else used adam.
    if chosen_model == "AlexNet":
        return torch.optim.RMSprop(model.parameters(), lr=1e-4)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def _run_epoch(model, loader, criterion, optimizer, is_googlenet, train_mode):
    """One pass over a DataLoader. Returns (mean_loss, accuracy)."""
    model.train() if train_mode else model.eval()
    total_loss = 0.0; correct = 0; total = 0
    torch.set_grad_enabled(train_mode)
    for x, y in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        if train_mode:
            optimizer.zero_grad()
        if is_googlenet and train_mode:
            out, aux1, aux2 = model(x)
            loss = criterion(out, y) + 0.3 * criterion(aux1, y) + 0.3 * criterion(aux2, y)
        else:
            out = model(x)
            loss = criterion(out, y)
        if train_mode:
            loss.backward(); optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)
    return total_loss / max(total, 1), correct / max(total, 1)


def make_fit_model(chosen_model, number_epoch, num_batch_size, num_early_patience, num_test):
    global model_trained, network, epochs, batch_size, early_patience, past_param_model
    history_train = {'accuracy': [], 'loss': []}
    history_val = {'accuracy': [], 'loss': []}
    history_test = {'accuracy': [], 'loss': []}

    error_text.set('')
    if len(total_image_ds) == 0:
        error_text.set(er_train_without_ds_text); return
    if not check_fit_param(number_epoch, num_batch_size, num_early_patience):
        return
    if chosen_model == "None":
        error_text.set(er_no_model_specified_text); return

    n_test = validate_num_test(num_test)
    use_greyscale = toggle_grey_scale.get()
    is_googlenet = (chosen_model == "GoogleNet")
    criterion = nn.CrossEntropyLoss()

    print("------------------------ MAKE AND FIT MODEL ------------------------")
    start_all_time = time.time()
    for i in range(n_test):
        print("-------- start: test num ", i, " --------")
        make_set_ds()
        status_model_text.set('Model: working')

        # -- build model --
        network = _build_network(chosen_model)
        if network is None:
            error_text.set(er_no_model_specified_text); return
        optimizer = _optimizer_for(chosen_model, network)
        past_param_model.update({"type_model": chosen_model, "epochs": epochs,
                                 "batch_size": batch_size, "early_patience": early_patience})

        # -- data loaders --
        train_loader = DataLoader(ImageFolderDataset(train_image, train_label, use_greyscale),
                                  batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(ImageFolderDataset(val_img, val_label, use_greyscale),
                                batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(ImageFolderDataset(test_image, test_label, use_greyscale),
                                 batch_size=batch_size, shuffle=False, num_workers=0)

        # -- training loop with early stopping + best-weights checkpoint --
        print("------------------------ fit model ------------------------")
        gpu_memory_usage()
        os.makedirs(path_check_point_model, exist_ok=True)
        ckpt_path = os.path.join(path_check_point_model, f"weight_{chosen_model}.pt")

        best_val_loss = float('inf'); best_state = None; patience_counter = 0
        hist = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
        start_time = time.time()
        for epoch in range(epochs):
            tr_loss, tr_acc = _run_epoch(network, train_loader, criterion, optimizer, is_googlenet, True)
            va_loss, va_acc = _run_epoch(network, val_loader, criterion, optimizer, is_googlenet, False)
            hist['loss'].append(tr_loss); hist['accuracy'].append(tr_acc)
            hist['val_loss'].append(va_loss); hist['val_accuracy'].append(va_acc)
            print(f"Epoch {epoch+1}/{epochs} - loss: {tr_loss:.4f} acc: {tr_acc:.4f} "
                  f"- val_loss: {va_loss:.4f} val_acc: {va_acc:.4f}")

            if va_loss < best_val_loss:                 # checkpoint best model
                best_val_loss = va_loss
                best_state = copy.deepcopy(network.state_dict())
                torch.save(best_state, ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_patience:  # early stopping
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:                      # restore best weights
            network.load_state_dict(best_state)
        end_time = time.time()
        print("Time for training the model: ", converti_secondi(end_time - start_time), " - test number: ", i)

        model_trained = True
        status_model_text.set('Model: trained')

        if n_test == 1:
            plot_fit_result(hist, 0)
        else:
            history_train['accuracy'].append(hist['accuracy'][-1])
            history_train['loss'].append(hist['loss'][-1])
            history_val['accuracy'].append(hist['val_accuracy'][-1])
            history_val['loss'].append(hist['val_loss'][-1])

        # -- evaluate on test set --
        test_loss, test_acc = _run_epoch(network, test_loader, criterion, optimizer, is_googlenet, False)
        if n_test == 1:
            plot_fit_result({'loss': test_loss, 'accuracy': test_acc}, 1)
            compute_confusion_matrix(network, test_loader, classes)
        else:
            history_test['loss'].append(test_loss)
            history_test['accuracy'].append(test_acc)
        print("-------- end: test num ", i, " --------")

    end_all_time = time.time()
    if n_test > 1:
        plot_accuracy_and_loss(history_train, history_val, history_test)
        print_average_metrics(history_train, history_val, history_test)
    print("Time for training all the tests: ", converti_secondi(end_all_time - start_all_time))
    print("------------------------------------------------")


def model_evaluate(param):
    error_text.set('')
    if not model_trained:
        error_text.set(er_eval_without_model_text); return
    use_greyscale = toggle_grey_scale.get()
    is_googlenet = isinstance(network, GLNet.GoogLeNet)
    criterion = nn.CrossEntropyLoss()

    if param == "test":
        if len(test_image) == 0 or len(test_label) == 0:
            error_text.set(er_eval_without_model_text); return
        loader = DataLoader(ImageFolderDataset(test_image, test_label, use_greyscale),
                            batch_size=batch_size, shuffle=False)
    elif param == "extern":
        if len(test_image_ext) == 0 or len(test_label_ext) == 0:
            error_text.set(er_no_ext_ds_text); return
        loader = DataLoader(ImageFolderDataset(test_image_ext, test_label_ext, use_greyscale),
                            batch_size=batch_size, shuffle=False)
    else:
        return

    test_loss, test_acc = _run_epoch(network, loader, criterion, None, is_googlenet, False)
    plot_fit_result({'loss': test_loss, 'accuracy': test_acc}, 1)
    compute_confusion_matrix(network, loader, classes)


def load_saved_model(model_name):
    global network, model_trained
    error_text.set('')
    if model_name:
        save_path = os.path.join(path_dir_model, model_name)
        if os.path.exists(save_path):
            # a full-model save (torch.save(model)) is loaded here; for a
            # state_dict save the architecture must be rebuilt first.
            network = torch.load(save_path, map_location=DEVICE, weights_only=False)
            network.to(DEVICE); network.eval()
            model_trained = True
            status_model_text.set('Model: trained')
        else:
            error_text.set(er_load_model_unknown_text)
    else:
        error_text.set(er_load_model_text)


def save_model(model_name):
    error_text.set('')
    if model_name:
        os.makedirs(path_dir_model, exist_ok=True)
        save_path = os.path.join(path_dir_model, model_name)
        torch.save(network, save_path)      # save the whole model (arch + weights)
    else:
        error_text.set(er_save_model_text)


def _preprocess_single(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    if img.shape[:2] != (img_height, img_width):
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)


def predict():
    if index_image_visualized != -1 and model_trained:
        error_text.set('')
        network.eval()
        if label_image_text.get() != '':
            if len(test_image) == 0 or len(test_label) == 0:
                path = total_image_ds[index_image_visualized]
            else:
                path = test_image[index_image_visualized]
            tensor = _preprocess_single(path)
            if tensor is None:
                return
            with torch.no_grad():
                logits = network(tensor)
            classify_text.set(': ' + str(classes[int(logits.argmax(1).item())]))
        elif label_ext_image_text.get() != '':
            tensor = _preprocess_single(test_image_ext[index_image_visualized])
            if tensor is None:
                return
            with torch.no_grad():
                logits = network(tensor)
            classify_ext_text.set(': ' + str(classes[int(logits.argmax(1).item())]))
    else:
        error_text.set(er_predict_text)
# ------------------------------------ end: methods for CNN model ------------------------------------


# ------------------------------------ start: utility method ------------------------------------
def check_past_model(type_model):
    return (past_param_model["type_model"] == type_model and
            past_param_model["epochs"] == epochs and
            past_param_model["batch_size"] == batch_size and
            past_param_model["early_patience"] == early_patience)


def plot_accuracy_and_loss(train_hist, val_hist, test_hist):
    runs = range(1, len(train_hist['accuracy']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(runs, train_hist['accuracy'], label='Train Accuracy')
    plt.plot(runs, val_hist['accuracy'], label='Validation Accuracy')
    plt.plot(runs, test_hist['accuracy'], label='Test Accuracy')
    plt.xlabel('Run'); plt.ylabel('Accuracy'); plt.title('Accuracy per run')
    plt.legend(); plt.grid(True); plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(runs, train_hist['loss'], label='Train Loss')
    plt.plot(runs, val_hist['loss'], label='Validation Loss')
    plt.plot(runs, test_hist['loss'], label='Test Loss')
    plt.xlabel('Run'); plt.ylabel('Loss'); plt.title('Loss per run')
    plt.legend(); plt.grid(True); plt.show()


def print_average_metrics(train_hist, val_hist, test_hist):
    def avg(lst): return round(float(np.mean(lst)), 4)
    print("\nMean:")
    print(f"Train     -> Accuracy: {avg(train_hist['accuracy'])}, Loss: {avg(train_hist['loss'])}")
    print(f"Validation-> Accuracy: {avg(val_hist['accuracy'])}, Loss: {avg(val_hist['loss'])}")
    print(f"Test      -> Accuracy: {avg(test_hist['accuracy'])}, Loss: {avg(test_hist['loss'])}")


def plot_fit_result(arc, mode):
    result_dict = {}
    if mode == 0:
        result_dict["loss (training set)"] = arc["loss"]
        result_dict["accuracy (training set)"] = arc["accuracy"]
        if arc.get("val_loss") is not None:
            result_dict["loss (validation set)"] = arc["val_loss"]
            result_dict["accuracy (validation set)"] = arc["val_accuracy"]
    elif mode == 1:
        result_dict["loss (test set)"] = arc["loss"]
        result_dict["accuracy (test set)"] = arc["accuracy"]
    for k, v in result_dict.items():
        plot(k, v)


def plot(title, value_list):
    if not isinstance(value_list, (list, tuple, np.ndarray)):
        value_list = [value_list]
    fig = plt.figure()
    fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(value_list, 'o-b')
    plt.title(str(title)); plt.xlabel("# Epochs"); plt.ylabel("Value")
    plt.show()


def compute_confusion_matrix(model, loader, classes):
    """Confusion matrix over a DataLoader (rows = real class, cols = predicted)."""
    model.eval()
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            for t, p in zip(y, preds):
                conf_matrix[int(t)][int(p)] += 1

    total = max(np.sum(conf_matrix), 1)
    conf_perc = [[f" ({(conf_matrix[i][j] / total) * 100:.2f}%)"
                  for j in range(conf_matrix.shape[1])] for i in range(conf_matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{conf_matrix[i][j]}{conf_perc[i][j]}", va='center', ha='center', fontsize=12)
    plt.xlabel("Predicted", fontsize=14); plt.ylabel("Actual", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()


def GPU_check():
    print("-------------------- PYTORCH VERSION --------------------")
    print(torch.__version__)
    print("-------------------- AVAILABLE HW DEVICES --------------------")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("Using device:", DEVICE)
    else:
        print("Using device: CPU")
    print("------------------------------------------------------------")


def gpu_memory_usage():
    print("-------------------- VRAM USAGE --------------------")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"VRAM allocated: {used:.2f} MB - reserved: {reserved:.2f} MB")
    print("------------------------------------------------------------")


def converti_secondi(sec):
    ore = sec // 3600
    minuti = (sec // 60) % 60
    secondi = sec % 60
    return f"{ore:.0f}h {minuti:.0f}m {secondi:.0f}s"
# ------------------------------------ end: utility method ------------------------------------


# ------------------------------------ main ------------------------------------
if __name__ == "__main__":
    window.title("GenCNNClassifier (PyTorch)")
    window.geometry(str(window_width) + 'x' + str(window_height))
    window.resizable(False, False)

    GPU_check()
    current_view_to_visualise()
    window.mainloop()

# ------------------------------------ Notes ------------------------------------
# -- Note 0 --
# The "classes" array is built at load time from the dataset sub-folders, so the
# number and names of classes are not fixed a priori: the same GUI can therefore
# be used on any image dataset organised as one sub-folder per class (pneumonia,
# satellite imagery, etc.), which is the intended cross-dataset testing use case.
