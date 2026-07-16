# -*- coding: utf-8 -*-
"""
@author: Alessandro Diana

explanation: PyTorch port of the class reproducing the GoogLeNet model
             (22 layers, two auxiliary classifiers).

description: network description at the end of the file.

Port notes (Keras -> PyTorch):
    - In the TensorFlow version the inception module used 1x1 kernels for every
      branch. Here the canonical GoogLeNet inception is reproduced faithfully to
      the original TF file: branch kernels are also 1x1 (matching the original
      code exactly). The 3x3/5x5 "same" padding is handled natively.
    - forward() returns a tuple (main, aux1, aux2) during training, mirroring the
      three-output Keras model. In eval mode it returns only the main logits.
    - Output layers return logits (no softmax): CrossEntropyLoss folds it in.
      The total loss is main + 0.3*aux1 + 0.3*aux2, as in the TF loss_weights.
    - Minimum input size: because of the auxiliary classifiers' 5x5 average
      pooling, the input resolution must be reasonably large (>= ~128x128;
      224x224 is the intended size). This matches the original TF model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Canonical inception module (four parallel paths, channel concat)."""

    def __init__(self, in_channels, fil_1x1, fil_1x1_3x3, fil_3x3,
                 fil_1x1_5x5, fil_5x5, fil_m_pool):
        super().__init__()
        self.path1 = nn.Conv2d(in_channels, fil_1x1, kernel_size=1, padding='same')

        self.path2_reduce = nn.Conv2d(in_channels, fil_1x1_3x3, kernel_size=1, padding='same')
        self.path2_conv = nn.Conv2d(fil_1x1_3x3, fil_3x3, kernel_size=1, padding='same')

        self.path3_reduce = nn.Conv2d(in_channels, fil_1x1_5x5, kernel_size=1, padding='same')
        self.path3_conv = nn.Conv2d(fil_1x1_5x5, fil_5x5, kernel_size=1, padding='same')

        self.path4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_conv = nn.Conv2d(in_channels, fil_m_pool, kernel_size=1, padding='same')

        self.out_channels = fil_1x1 + fil_3x3 + fil_5x5 + fil_m_pool

    def forward(self, x):
        p1 = F.relu(self.path1(x))
        p2 = F.relu(self.path2_conv(F.relu(self.path2_reduce(x))))
        p3 = F.relu(self.path3_conv(F.relu(self.path3_reduce(x))))
        p4 = F.relu(self.path4_conv(self.path4_pool(x)))
        return torch.cat([p1, p2, p3, p4], dim=1)


class _AuxClassifier(nn.Module):
    """Auxiliary classifier head. Uses lazy Linear to avoid hard-coding the
    flattened size (depends on input resolution)."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, padding='same')
        self.fc1 = nn.LazyLinear(1024)
        self.drop = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)          # logits


class GoogLeNet(nn.Module):

    def __init__(self, class_number, img_width=224, img_height=224, img_channel=3):
        super().__init__()
        self.num_classes = class_number
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel

    def make_model(self):
        C = self.img_channel
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding='same'), nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding='same'), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        # block feeding aux_1
        self.inc3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inc3b = InceptionModule(self.inc3a.out_channels, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.inc4a = InceptionModule(self.inc3b.out_channels, 192, 96, 208, 16, 48, 64)

        self.aux1 = _AuxClassifier(self.inc4a.out_channels, self.num_classes)

        # block feeding aux_2
        self.inc4b = InceptionModule(self.inc4a.out_channels, 160, 112, 224, 24, 64, 64)
        self.inc4c = InceptionModule(self.inc4b.out_channels, 128, 128, 256, 24, 64, 64)
        self.inc4d = InceptionModule(self.inc4c.out_channels, 112, 144, 288, 32, 64, 64)

        self.aux2 = _AuxClassifier(self.inc4d.out_channels, self.num_classes)

        # final block
        self.inc4e = InceptionModule(self.inc4d.out_channels, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2)
        self.inc5a = InceptionModule(self.inc4e.out_channels, 256, 160, 320, 32, 128, 128)
        self.inc5b = InceptionModule(self.inc5a.out_channels, 384, 192, 384, 48, 128, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.fc_out = nn.Linear(self.inc5b.out_channels, self.num_classes)

        # materialise LazyLinear layers with a dummy forward
        self.train()
        with torch.no_grad():
            dummy = torch.zeros(2, self.img_channel, self.img_height, self.img_width)
            self.forward(dummy)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GoogLeNet built - trainable params: {n_params:,}")

    def forward(self, x):
        x = self.stem(x)
        x = self.inc3a(x)
        x = self.inc3b(x)
        x = self.pool3(x)
        x = self.inc4a(x)

        aux1 = self.aux1(x) if self.training else None

        x = self.inc4b(x)
        x = self.inc4c(x)
        x = self.inc4d(x)

        aux2 = self.aux2(x) if self.training else None

        x = self.inc4e(x)
        x = self.pool4(x)
        x = self.inc5a(x)
        x = self.inc5b(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        out = self.fc_out(x)        # logits

        if self.training:
            return out, aux1, aux2
        return out

    def return_model(self):
        return self


"""
brief description:
    GoogLeNet won ILSVRC-2014. Published in "Going Deeper with Convolutions", it
    reaches a depth of 22 layers using inception modules (1x1, 3x3, 5x5 conv and
    3x3 max pool executed in parallel and concatenated), 1x1 convolutions to keep
    the parameter count down, and global average pooling instead of large fully
    connected layers. Two auxiliary classifiers are attached mid-network to help
    gradient flow during training; their losses are added to the main loss with
    weight 0.3 each. Input size 224x224 RGB, ReLU activations throughout.
"""
