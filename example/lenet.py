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

def test(model, test_dataloader, criterion, is_my=False):
    model.eval()
    
    loss = 0
    accuracy = 0
    round_num = 0
    for (x,y) in test_dataloader:
        if is_my:
            x = Tensor(x.detach().cpu().numpy(), autograd=True)
            y = Tensor(y.detach().cpu().numpy(), autograd=True)
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
    
    def __call__(self, x):
        return self.forward(x)
    
def train(epochs, train_dataloader, test_dataloader, model, criterion, optimzer, is_my=False):
    for epoch in range(epochs):
        loss = 0
        accuracy = 0
        counter = 0
        start = time.time()
        
        for x,y in train_dataloader:
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
            counter += 1
       
        # do test
        test_loss, test_acc = test(model, test_dataloader, criterion, is_my)
        # end time counter
        end = time.time()
        print('in epoch %d, dura %.4f sec, train loss: %.4f, train acc: %.4f, test loss: %.4f, test acc: %.4f' % \
            (epoch, end-start, loss/counter, accuracy/counter, test_loss, test_acc))
    
def pytorch_train(batch_size, epochs, alpha, classes_num, mnist_ds_path):
    # model
    lenet = LeNet(classes_num)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # no momentum means poor performance(may drop 3 percent), momentum=0.9 is good enough
    sgd = t.optim.SGD(lenet.parameters(), lr=alpha, momentum=0.9)
    # data
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
    epochs = 2
    alpha = 0.001
    classes_num = 10
    mnist_ds_path = 'datasets'

    # in 2 epochs, it can reach
    # in epoch 0, dura 17.3718 sec, train loss: 1.4284, train acc: 0.5847, test loss: 0.2987, test acc: 0.9130
    # in epoch 1, dura 16.7949 sec, train loss: 0.2464, train acc: 0.9236, test loss: 0.1629, test acc: 0.9519
    pytorch_train(batch_size, epochs, alpha, classes_num, mnist_ds_path)
    # in 2 epochs, it can reach
    # in epoch 0, dura 11579.6827 sec, train loss: 0.2233, train acc: 0.9294, test loss: 0.0703, test acc: 0.9769
    # in epoch 1, dura 11416.8372 sec, train loss: 0.0678, train acc: 0.9789, test loss: 0.0492, test acc: 0.9845
    # well, the speed is a big problem
    # after rewrite the core function(conv2d, max pool 2d) in cpp, the speed is as followings:
    # in epoch 0, dura 124.6791 sec, train loss: 0.2207, train acc: 0.9299, test loss: 0.0691, test acc: 0.9776
    # in epoch 1, dura 124.7624 sec, train loss: 0.0656, train acc: 0.9799, test loss: 0.0456, test acc: 0.9854
    my_train(batch_size, epochs, alpha, classes_num, mnist_ds_path)
    # I have find that if I implement conv\max pool in cpp by double not float, the accuracy will a little higher
    
    # in float mode, the result is as followings:
    # torch:
    # we have 1875 train samples, and 313 test samples
    # in epoch 0, dura 17.3973 sec, train loss: 1.0882, train acc: 0.6762, test loss: 0.2792, test acc: 0.9189
    # in epoch 1, dura 16.5384 sec, train loss: 0.2326, train acc: 0.9274, test loss: 0.1601, test acc: 0.9471
    # in epoch 2, dura 16.6213 sec, train loss: 0.1475, train acc: 0.9544, test loss: 0.1289, test acc: 0.9578
    # in epoch 3, dura 16.2555 sec, train loss: 0.1106, train acc: 0.9656, test loss: 0.1104, test acc: 0.9627
    # in epoch 4, dura 16.5282 sec, train loss: 0.0912, train acc: 0.9709, test loss: 0.0790, test acc: 0.9743
    # my implementation
    # we have 1875 train samples, and 313 test samples
    # in epoch 0, dura 121.5823 sec, train loss: 0.2940, train acc: 0.9100, test loss: 0.0770, test acc: 0.9751
    # in epoch 1, dura 121.3367 sec, train loss: 0.0710, train acc: 0.9777, test loss: 0.0775, test acc: 0.9750
    # in epoch 2, dura 121.4589 sec, train loss: 0.0503, train acc: 0.9843, test loss: 0.0512, test acc: 0.9824
    # in epoch 3, dura 121.5643 sec, train loss: 0.0382, train acc: 0.9879, test loss: 0.0478, test acc: 0.9845
    # in epoch 4, dura 121.3851 sec, train loss: 0.0312, train acc: 0.9902, test loss: 0.0364, test acc: 0.9872