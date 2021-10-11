import os
import sys
import time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep)

import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch as t
from torch import nn
from torch.nn import functional as F

from framework.cnn import Conv2d
from framework.dropout import Dropout
from framework.layer import MaxPool2d, LinearLayer, Sequential
from framework.activation import Relu
from framework.loss import CrossEntropyLoss
from framework.optimizer import SGD
from framework.tensor import Tensor


class VGG16(Sequential):
    def __init__(self, classes_num):
        super(VGG16, self).__init__()
  
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = MaxPool2d(kernel_size=2)

        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = MaxPool2d(kernel_size=2)

        self.conv5 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = MaxPool2d(kernel_size=2)

        self.conv8 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = MaxPool2d(kernel_size=2)

        self.conv11 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool5 = MaxPool2d(kernel_size=2)

        self.linear1 = LinearLayer(512*7*7, 4096)
        self.linear2 = LinearLayer(4096, 4096)
        self.classifier = LinearLayer(4096, classes_num)
        self.relu = Relu()
        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2(x)
        
        x = x.view((x.shape[0], -1))
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        pred = self.classifier(x)
        
        return pred
