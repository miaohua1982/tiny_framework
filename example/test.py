import os
import sys

from numpy.lib.npyio import save
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep)

from framework.tensor import Tensor
from framework.rnn import RNN_Model, Lstm_Model
from framework.layer import Sequential, MaxPool2d, LinearLayer
from framework.cnn import Conv2d
from framework.activation import Relu
from framework.utils import save_model, load_model, all
import numpy as np


lr = LinearLayer(4096, 1000, bias=False)
print(lr.get_input_features())
print(lr)