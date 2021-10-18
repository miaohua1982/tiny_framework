import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep+os.path.pardir+os.path.sep)

from framework.cnn import Conv2d
from framework.dropout import Dropout
from framework.layer import Sequential
from framework.maxpool import MaxPool2d
from framework.linear import LinearLayer
from framework.activation import Relu


class AlexNet(Sequential):
    def __init__(self, classes_num):
        super(AlexNet, self).__init__()
  
        self.conv1 = Conv2d(3, 96, kernel_size=11, stride=4, padding=2)  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
        self.max_pool1 = MaxPool2d(kernel_size=3, stride=2)              # output[96, 27, 27] kernel_num为原论文一半
        self.conv2 = Conv2d(96, 256, kernel_size=5, padding=2)           # output[256, 27, 27]
        self.max_pool2 = MaxPool2d(kernel_size=3, stride=2)              # output[256, 13, 13]
        self.conv3 = Conv2d(256, 384, kernel_size=3, padding=1)          # output[384, 13, 13]
        self.conv4 = Conv2d(384, 384, kernel_size=3, padding=1)          # output[384, 13, 13]
        self.conv5 = Conv2d(384, 256, kernel_size=3, padding=1)          # output[256, 13, 13]
        self.max_pool3 = MaxPool2d(kernel_size=3, stride=2)              # output[256, 6, 6]
        self.relu = Relu(inplace=True)

        self.linear1 = LinearLayer(256 * 6 * 6, 256)
        self.linear2 = LinearLayer(256, 256)
        self.classifier = LinearLayer(256, classes_num)
        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.max_pool3(x)

        x = x.view((x.shape[0], -1))
        
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        pred = self.classifier(x)
        
        return pred