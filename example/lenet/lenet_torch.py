from torch import nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self, classes_num):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(16*7*7, 512)
        self.linear2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, classes_num)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        pred = self.classifier(x)
        
        return pred
