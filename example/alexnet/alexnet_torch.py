import torch as t
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):   
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)  # input[3, 224, 224]  output[48, 55, 55] 自动舍去小数点后
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)                   # output[48, 27, 27] kernel_num为原论文一半
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)           # output[128, 27, 27]
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)                   # output[128, 13, 13]
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)          # output[192, 13, 13]
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)          # output[192, 13, 13]
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)          # output[128, 13, 13]
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)                   # output[128, 6, 6]
        self.relu = nn.ReLU(inplace=True)
        
        self.classifier = nn.Sequential(
            #全链接
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max1(x)

        x = self.relu(self.conv2(x))
        x = self.max2(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.max3(x)

        x = x.view((x.shape[0], -1))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  #正态分布赋值
                nn.init.constant_(m.bias, 0)
