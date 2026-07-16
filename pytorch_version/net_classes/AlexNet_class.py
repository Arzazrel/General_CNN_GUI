# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: PyTorch port of the class reproducing the AlexNet model
             (batch-normalised variant, as in the original TensorFlow code).

description: network description at the end of the file.

Port notes: see IfritNet_class.py. Output layer returns logits (no softmax),
since nn.CrossEntropyLoss applies log-softmax internally.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, class_number, img_width=224, img_height=224, img_channel=3):
        super().__init__()
        self.num_classes = class_number
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.features = None
        self.classifier = None

    def make_model(self):
        C = self.img_channel
        self.features = nn.Sequential(
            # 1st conv layer (with max pool)
            nn.Conv2d(C, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(96),
            # 2nd conv layer (with max pool)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(256),
            # 3rd conv layer (no max pool)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            # 4th conv layer (no max pool)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            # 5th conv layer (with max pool)
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(256),
        )

        flat = self._infer_flatten_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.BatchNorm1d(4096),
            nn.Linear(4096, self.num_classes),      # logits (no softmax)
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"AlexNet built - trainable params: {n_params:,}")

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def _infer_flatten_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.img_channel, self.img_height, self.img_width)
            out = self.features(dummy)
        return int(out.numel())

    def return_model(self):
        return self


"""
brief description:
    AlexNet is a CNN designed by Alex Krizhevsky and Ilya Sutskever under the
    supervision of Geoffrey Hinton. It won the ImageNet Large Scale Visual
    Recognition Challenge in 2012 and marked a breakthrough in image
    classification. It has eight layers: five convolutional (the first two with
    max-pooling) and three fully connected. The network uses ReLU activations
    and, in this variant, batch normalisation in place of the original local
    response normalisation. The final layer uses softmax (here folded into the
    CrossEntropyLoss).
"""
