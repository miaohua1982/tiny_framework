import os
import sys
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

def data_loader(ds_path, batch_size):
    train_ds = tv.datasets.MNIST(root=ds_path, train=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  #60000

    test_ds = tv.datasets.MNIST(root=ds_path, train=False, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)    #10000

    print('we have %d train samples, and %d test samples' % (len(train_dataloader), len(test_dataloader)))
    
    return train_dataloader, test_dataloader

def calc_accu(pred, gt):
    final = pred.argmax(dim=1)
    return (final == gt).float().mean()

def test(model, test_dataloader, criterion):
    model.eval()
    
    loss = 0
    accuracy = 0
    round_num = 0
    for (x,y) in test_dataloader:
        pred = model(x)
        
        cur_loss = criterion(pred, y)
        cur_accuracy = calc_accu(pred, y)
        
        loss += cur_loss.item()
        accuracy += cur_accuracy.item()
        
        round_num += 1
        
    model.train()
    
    return loss/round_num, accuracy/round_num

class LeNet(nn.Module):
    def __init__(self, classes_num):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3), stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d((2,2))
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

class MyLeNet(Sequential):
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
    
def train(epochs, train_dataloader, test_dataloader, model, criterion, optimzer, is_my=False):
    for epoch in range(epochs):
        loss = 0
        accuracy = 0
        round_num = 0
        for idx, (x,y) in enumerate(train_dataloader):
            if is_my:
                x = Tensor(x.detach().cpu().numpy(), autograd=True)
                y = Tensor(y.detach().cpu().numpy(), autograd=True)

            pred = model(x)
            
            optimzer.zero_grad()
            
            cur_loss = criterion(pred, y)
            cur_accuracy = calc_accu(pred, y)
            
            cur_loss.backward()
            optimzer.step()
            
            loss += cur_loss.item()
            accuracy += cur_accuracy.item()
            
            round_num += 1
        
        # do test
        test_loss, test_acc = test(model, test_dataloader, criterion)
        print('in epoch %d, train loss: %.4f, train acc: %.4f, test loss: %.4f, test acc: %.4f' % \
            (epoch, loss/round_num, accuracy/round_num, test_loss, test_acc))
    
def pytorch_train(batch_size, epochs, alpha, classes_num, mnist_ds_path):
    # model
    lenet = LeNet(classes_num)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # no momentum means poor performance(may drop 3 percent), momentum=0.9 is good enough
    sgd = t.optim.SGD(lenet.parameters(), lr=alpha, momentum=0.9)
    train_dataloader, test_dataloader = data_loader(mnist_ds_path, batch_size)
    # train
    train(epochs, train_dataloader, test_dataloader, lenet, criterion, sgd)

def my_train(batch_size, epochs, alpha, classes_num, mnist_ds_path):
    # model
    lenet = MyLeNet(classes_num)
    # loss function
    criterion = CrossEntropyLoss()
    # optimizer
    sgd = SGD(lenet.get_parameters(), lr=alpha)
    # data
    train_dataloader, test_dataloader = data_loader(mnist_ds_path, batch_size)
    # train
    train(epochs, train_dataloader, test_dataloader, lenet, criterion, sgd, True)

if __name__ == '__main__':
    batch_size = 32
    epochs = 35
    alpha = 0.001
    classes_num = 10
    mnist_ds_path = 'datasets'

    #pytorch_train(batch_size, epochs, alpha, classes_num, mnist_ds_path)
    my_train(batch_size, epochs, alpha, classes_num, mnist_ds_path)