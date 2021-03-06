import os
import sys
import time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep+os.path.pardir+os.path.sep)

from framework.tensor import Tensor
from framework.loss import CrossEntropyLoss
from framework.optimizer import SGD, Adam
from framework.utils import save_model, load_model
from alexnet_torch import AlexNet as AlexNet_T
from alexnet_tiny import AlexNet

import torch as t
from torchvision import transforms
import torchvision
from torch import optim
from torch import nn
from skimage import transform as sktsf

"""
1. load CIFAR10 datasets, 3*32*32 for one picture
"""
ds_path = '/home/miaohua/Documents/Datasets'
batch_size = 16
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root=ds_path, train=True, download=False, transform=transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root=ds_path, train=False, download=False, transform=transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def accu(pred, target):
    p = pred.argmax(dim=1)
    return (p == target).float().mean()

def model_test(net, criterion, use_tiny_framework):
    running_loss = 0.0
    running_acc = 0.0
    for data in testloader:
        inputs, labels = data  # labels: [batch_size, 1]
        
        inputs = sktsf.resize(inputs.detach().cpu().numpy(), (batch_size, 3, 224, 224), mode='reflect', anti_aliasing=False)
        if use_tiny_framework:
            inputs = Tensor(inputs, autograd=True)
            labels = Tensor(labels.detach().cpu().numpy(), autograd=True)
        else:
            inputs = t.from_numpy(inputs)
        
        outputs = net(inputs)  # outputs: [batch_size, 10]
        # loss
        loss = criterion(outputs, labels)
        # acc
        acc = accu(outputs, labels)
        # ??????loss
        running_loss += loss.item()
        running_acc += acc.item()

    print('In test set loss: %.5f, accu: %.5f' % (running_loss/len(testloader), running_acc/len(testloader)))

def model_train(classes_num, use_tiny_framework, spot_plot, model_path):
    epochs = 10  # ????????????
    learning_rate = 1e-4  # ?????????

    if use_tiny_framework:
        if model_path is None:
            net = AlexNet(classes_num)
        else:
            net = load_model(model_path)

        criterion = CrossEntropyLoss()
        # optimizer = SGD(net.get_parameters(), lr=learning_rate)
        optimizer = Adam(net.get_parameters(), lr=learning_rate)
    else:
        if model_path is None:
            net = AlexNet_T(classes_num)
        else:
            net = t.load(model_path)

        criterion = nn.CrossEntropyLoss() # ???????????????
        optimizer = optim.Adam(net.parameters(), lr=learning_rate) # Adam?????????


    for epoch in range(epochs):  # ??????
        running_loss = 0.0
        running_acc = 0.0

        total_acc = 0.0
        total_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data  # labels: [batch_size, 1]

            inputs = sktsf.resize(inputs.detach().cpu().numpy(), (batch_size, 3, 224, 224), mode='reflect', anti_aliasing=False)
            if use_tiny_framework:
                inputs = Tensor(inputs, autograd=True)
                labels = Tensor(labels.detach().cpu().numpy(), autograd=True)
            else:
                inputs = t.from_numpy(inputs)
            
            # ???????????????
            optimizer.zero_grad()

            outputs = net(inputs)  # outputs: [batch_size, 10]
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = accu(outputs, labels)
            # ??????loss
            running_loss += loss.item()
            running_acc += acc.item()
            total_loss += loss.item()
            total_acc += acc.item()

            if i % 20 == 19 and spot_plot:  # print loss every 20 mini batch
                print('[%s] [%d, %5d] loss: %.5f, accu: %.5f' %
                      (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch+1, i+1, running_loss/20.0, running_acc/20.0))
                running_loss = 0.0
                running_acc = 0.0
        print('[%d, %5d] loss: %.5f, accu: %.5f' % (epoch+1, epochs, total_loss/len(trainloader), total_acc/len(trainloader)))
        
        model_test(net, criterion, use_tiny_framework)
        
        if use_tiny_framework:
            save_model(net, 'model_storage/vgg16_%s_%d.pkl' % (time.strftime('%Y-%m-%d',time.localtime(time.time())), epoch+1))
        else:
            t.save(net.state_dict(), 'model_storage/vgg16_%s_%d_t.pkl' % (time.strftime('%Y-%m-%d',time.localtime(time.time())), epoch+1))

    print('Finished Training')


# ?????????????????????CIFAR10?????????????????? 32*32*3
if __name__ == '__main__':
    classes_num = 10
    use_tiny_framework = False
    spot_plot = False
    model_path = None
    model_train(classes_num, use_tiny_framework, spot_plot, model_path)