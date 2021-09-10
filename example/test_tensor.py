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

# print tensor 
a = Tensor(np.array([1,2,3]))
print(a)

# print rnn model
word_embedding_size = 64
hidden_size = 64
vocab_size = 1024
model = RNN_Model(word_embedding_size, hidden_size, vocab_size)
print(model)

model = Lstm_Model(word_embedding_size, hidden_size, vocab_size)
print(model)

# print conv model (the lenet)
class LeNet(Sequential):
    def __init__(self, classes_num):
        super(LeNet, self).__init__()
  
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
        self.add(self.linear1)
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
    
    def __call__(self, x):
        return self.forward(x)

classes_num = 10
model = LeNet(classes_num)
print(model)

# load & save & compare
save_model(model, 'lenet.dmp')
model2 = load_model('lenet.dmp')
print(all(model2.conv1.kernel == model.conv1.kernel))
