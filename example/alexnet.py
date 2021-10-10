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
from framework.layer import MaxPool2d, LinearLayer, Sequential
from framework.activation import Relu
from framework.loss import CrossEntropyLoss
from framework.optimizer import SGD
from framework.tensor import Tensor


class AlexNet(Sequential):
    def __init__(self, classes_num):
        super(MyLeNet, self).__init__()
  
        self.conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = MaxPool2d(kernel_size=2)
        self.linear1 = LinearLayer(16*7*7, 512)
        self.linear2 = LinearLayer(512, 256)
        self.classifier = LinearLayer(256, classes_num)
        self.relu = Relu()

        self.add(self.conv1)
        self.add(self.relu)
        self.add(self.max_pool1)
        self.add(self.conv2)
        self.add(self.relu)
        self.add(self.max_pool2)
        self.add(self.linear1)
        self.add(self.relu)
        self.add(self.linear2)
        self.add(self.relu)        
        self.add(self.classifier)

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
