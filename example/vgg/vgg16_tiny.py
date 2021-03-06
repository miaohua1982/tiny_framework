import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep+os.path.pardir+os.path.sep)

import torchvision as tv

from framework.cnn import Conv2d
from framework.dropout import Dropout
from framework.layer import Sequential
from framework.maxpool import MaxPool2d
from framework.linear import LinearLayer
from framework.activation import Relu


class VGG16(Sequential):
    def __init__(self, classes_num):
        super(VGG16, self).__init__()
  
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = MaxPool2d(kernel_size=2)

        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = MaxPool2d(kernel_size=2)

        self.conv5 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.max_pool3 = MaxPool2d(kernel_size=2)

        self.conv8 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool4 = MaxPool2d(kernel_size=2)

        self.conv11 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.max_pool5 = MaxPool2d(kernel_size=2)

        self.linear1 = LinearLayer(512*1*1, 256)
        self.linear2 = LinearLayer(256, 256)
        self.classifier = LinearLayer(256, classes_num)
        self.relu = Relu()
        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max_pool2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.max_pool3(x)
        
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.max_pool4(x)

        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.max_pool5(x)

        x = x.view((x.shape[0], -1))
        
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        pred = self.classifier(x)
        
        return pred

def copy_weights_from_pretrained(model):
    map_layer_name = {'features.0.weight':'conv1', 'features.0.bias':'conv1', 'features.2.weight':'conv2', 'features.2.bias':'conv2',
                      'features.5.weight':'conv3', 'features.5.bias':'conv3',  'features.7.weight':'conv4', 'features.7.bias':'conv4',
                      'features.10.weight':'conv5', 'features.10.bias':'conv5', 'features.12.weight':'conv6', 'features.12.bias':'conv6', 'features.14.weight':'conv7', 'features.14.bias':'conv7',
                      'features.17.weight':'conv8', 'features.17.bias':'conv8', 'features.19.weight':'conv9', 'features.19.bias':'conv9', 'features.21.weight':'conv10', 'features.21.bias':'conv10',
                      'features.24.weight':'conv11', 'features.24.bias':'conv11', 'features.26.weight':'conv12', 'features.26.bias':'conv12', 'features.28.weight':'conv13', 'features.28.bias':'conv13'}
            
    pretrained_model = tv.models.vgg16(pretrained=True)  # ????????????????????????VGG16????????????
    pretrained_params = pretrained_model.state_dict()
    for idx, (k, v) in enumerate(map_layer_name.items()):
        if idx%2 == 0:
            model.attr_dict()[v].set_weight(pretrained_params[k].detach().cpu().numpy())
        else:
            model.attr_dict()[v].set_bias(pretrained_params[k].detach().cpu().numpy())