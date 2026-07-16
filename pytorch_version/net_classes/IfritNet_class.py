# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: PyTorch port of the class containing the various versions of the
             CNN (IfritNet) originally designed for the fire-detection problem.

description: network description at the end of the file (see Notes).

Port notes (Keras -> PyTorch):
    - Keras works in channels-last (H, W, C); PyTorch works in channels-first
      (C, H, W). Tensors fed to these modules must already be (N, C, H, W).
    - Keras "padding='valid'" == PyTorch padding=0.
    - Keras "padding='same'" (stride=1) is reproduced with PyTorch's native
      padding='same', available for stride=1 convolutions/poolings.
    - The output layer here does NOT apply softmax: PyTorch's
      nn.CrossEntropyLoss expects raw logits and applies log-softmax internally.
      For inference probabilities, apply torch.softmax on the logits explicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------ start: utility modules ------------------------------------

class InceptionModule(nn.Module):
    """
    Inception module: 1x1, 3x3, 5x5 convolutions and a 3x3 max pooling are
    executed in parallel and their outputs are concatenated along the channel
    axis. Faithful port of the Keras 'inception_mod' helper.

    NOTE: this mirrors the (arguably unusual) original design, where the 3x3 and
    5x5 branches use a 1x1 reduction convolution followed by a *1x1* convolution
    (kernel_size=(1,1)) rather than the canonical 3x3 / 5x5 kernel. This is kept
    identical on purpose so that the PyTorch model reproduces the exact same
    architecture (and parameter count) as the TensorFlow one.

    in_channels : number of input channels
    fil_1x1     : filters of the pure 1x1 branch
    fil_1x1_3x3 : filters of the 1x1 reduction before the "3x3" branch
    fil_3x3     : filters of the "3x3" branch (kernel kept 1x1 as in the original)
    fil_1x1_5x5 : filters of the 1x1 reduction before the "5x5" branch
    fil_5x5     : filters of the "5x5" branch (kernel kept 1x1 as in the original)
    fil_m_pool  : filters of the 1x1 conv after max pooling
    """

    def __init__(self, in_channels, fil_1x1, fil_1x1_3x3, fil_3x3,
                 fil_1x1_5x5, fil_5x5, fil_m_pool):
        super().__init__()

        # path 1: single 1x1 conv
        self.path1 = nn.Conv2d(in_channels, fil_1x1, kernel_size=1, padding='same')

        # path 2: 1x1 reduce -> conv (kernel 1x1 as in the original code)
        self.path2_reduce = nn.Conv2d(in_channels, fil_1x1_3x3, kernel_size=1, padding='same')
        self.path2_conv = nn.Conv2d(fil_1x1_3x3, fil_3x3, kernel_size=1, padding='same')

        # path 3: 1x1 reduce -> conv (kernel 1x1 as in the original code)
        self.path3_reduce = nn.Conv2d(in_channels, fil_1x1_5x5, kernel_size=1, padding='same')
        self.path3_conv = nn.Conv2d(fil_1x1_5x5, fil_5x5, kernel_size=1, padding='same')

        # path 4: 3x3 max pool (stride 1, same) -> 1x1 conv to reduce
        self.path4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_conv = nn.Conv2d(in_channels, fil_m_pool, kernel_size=1, padding='same')

        # number of output channels of the module (used to chain modules)
        self.out_channels = fil_1x1 + fil_3x3 + fil_5x5 + fil_m_pool

    def forward(self, x):
        p1 = F.relu(self.path1(x))

        p2 = F.relu(self.path2_reduce(x))
        p2 = F.relu(self.path2_conv(p2))

        p3 = F.relu(self.path3_reduce(x))
        p3 = F.relu(self.path3_conv(p3))

        p4 = self.path4_pool(x)
        p4 = F.relu(self.path4_conv(p4))

        return torch.cat([p1, p2, p3, p4], dim=1)   # concat on channel axis

# ------------------------------------ end: utility modules ------------------------------------


def _conv_out(size, kernel, stride, padding=0):
    """Helper to compute the spatial size after a conv/pool with 'valid'-like padding."""
    return (size - kernel + 2 * padding) // stride + 1


class IfritNet(nn.Module):
    """
    Implements the IfritNet models (4 versions) as an nn.Module.

    Usage mirrors the TensorFlow class API:
        net = IfritNet(num_classes, img_width, img_height, img_channel)
        net.make_model(version)      # builds the chosen architecture in place
        model = net.return_model()   # returns the nn.Module itself
    """

    def __init__(self, class_number, img_width=224, img_height=224, img_channel=3):
        super().__init__()
        self.num_classes = class_number
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.features = None            # convolutional part
        self.classifier = None          # dense part
        self._is_inception = False      # flag for the version-4 forward path
        self.version = None

    # ------------------------------------------------------------------
    # build the architecture. 'version_model' selects the IfritNet variant.
    # ------------------------------------------------------------------
    def make_model(self, version_model):
        self.version = version_model
        C = self.img_channel

        if version_model == 1:                      # see Note 1
            self.features = nn.Sequential(
                # 1st conv layer
                nn.Conv2d(C, 32, kernel_size=7, stride=3, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(32),
                # 2nd conv layer
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1), nn.BatchNorm2d(64),
                # 3rd conv layer
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1), nn.BatchNorm2d(128),
            )
            flat = self._infer_flatten_size()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.BatchNorm1d(128),
                nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(128, self.num_classes),   # logits (no softmax)
            )

        elif version_model == 2:                    # see Note 2
            self.features = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=7, stride=3, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1), nn.BatchNorm2d(64),
            )
            flat = self._infer_flatten_size()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.BatchNorm1d(128),
                nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(128, self.num_classes),
            )

        elif version_model == 3:                    # see Note 3 (lite version)
            self.features = nn.Sequential(
                nn.Conv2d(C, 16, kernel_size=7, stride=3, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0), nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1), nn.BatchNorm2d(32),
            )
            flat = self._infer_flatten_size()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.BatchNorm1d(64),
                nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(64, self.num_classes),
            )

        elif version_model == 4:                    # see Note 4 (GoogLeNet-inspired)
            self._is_inception = True
            # first conv layer: kernel 7, stride 2, padding 'same'
            self.stem_conv = nn.Conv2d(C, 16, kernel_size=7, stride=2, padding=3)
            self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2)      # 'valid' pool (padding 0)

            self.inc1 = InceptionModule(16, 16, 8, 32, 16, 64, 32)
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)          # 'valid'
            self.inc2 = InceptionModule(self.inc1.out_channels, 16, 8, 32, 16, 64, 32)
            self.inc3 = InceptionModule(self.inc2.out_channels, 32, 16, 64, 16, 32, 64)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)          # 'valid'
            self.inc4 = InceptionModule(self.inc3.out_channels, 64, 16, 128, 8, 16, 16)
            self.gap = nn.AdaptiveAvgPool2d(1)                          # GlobalAveragePooling2D

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.inc4.out_channels, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(64, self.num_classes),
            )
        else:
            raise ValueError(f"Unknown IfritNet version: {version_model}")

        self._print_summary()

    # ------------------------------------------------------------------
    # forward pass (dispatches between sequential and inception variants)
    # ------------------------------------------------------------------
    def forward(self, x):
        if self._is_inception:                      # version 4
            x = F.relu(self.stem_conv(x))
            x = self.stem_pool(x)
            x = self.inc1(x)
            x = self.pool1(x)
            x = self.inc2(x)
            x = self.inc3(x)
            x = self.pool2(x)
            x = self.inc4(x)
            x = self.gap(x)
            return self.classifier(x)
        else:                                       # versions 1, 2, 3
            x = self.features(x)
            return self.classifier(x)

    # ------------------------------------------------------------------
    # utility: run a dummy tensor through the conv part to size the first Linear
    # ------------------------------------------------------------------
    def _infer_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.img_channel, self.img_height, self.img_width)
            out = self.features(dummy)
        return int(out.numel())

    def _print_summary(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"IfritNet version {self.version} built - trainable params: {n_params:,}")

    # API-compatibility helpers (mirror the TF class)
    def return_model(self):
        return self


"""
-------- Notes --------
-- Note 1 --
    Simple CNN consisting of 3 convolutional layers, cascaded with max pool and
    batch normalization, followed by 2 fully connected layers before the output.

-- Note 2 --
    CNN inspired by AlexNet: 4 convolutional layers, 2 fully connected layers
    with dropout, and the output layer. Max pool is present only after the first
    and fourth convolutional layers; normalization after all layers except the
    penultimate one.

-- Note 3 --
    "Lite" version of the second model, reducing the number of trainable weights
    to obtain a similar but much lighter and faster architecture.

-- Note 4 --
    CNN inspired by GoogLeNet: one convolutional layer with max pool, four
    inception modules, a fully connected layer with dropout, and the output.
    Max pool after the first and third inception modules, and global average
    pooling at the end. This is the version (Ifrit_4) used for cross-dataset
    testing (pneumonia, satellite imagery, ...).
"""
