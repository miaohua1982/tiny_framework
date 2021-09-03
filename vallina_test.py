from __future__ import absolute_import

import numpy as np
from framework.tensor import Tensor
from framework.layer import LinearLayer, Sequential, EmbeddingLayer
from framework.optimizer import SGD
from framework.loss import MSELoss, CrossEntropyLoss
from framework.activation import Tanh
from functools import wraps

np.random.seed(0)

alpha = 0.1
epoches = 10
orgw_0_1 = np.random.rand(2,3)
orgw_1_2 = np.random.rand(3,1)

def add_divider(comment):
    def add_divider_helper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            print('-'*32,comment,'-'*32)
            f(*args, **kwargs)
        return decorated
    return add_divider_helper
    

@add_divider(comment='all_manual')
def all_manual(epoches, alpha, orgw_0_1, orgw_1_2):
    weights_0_1 = orgw_0_1.copy()
    weights_1_2 = orgw_1_2.copy()
    data = np.array([[0,0],[0,1],[1,0],[1,1]])
    target = np.array([[0],[1],[0],[1]])

    for i in range(epoches):
        # Predict
        layer_1 = data.dot(weights_0_1)
        layer_2 = layer_1.dot(weights_1_2)
        
        # Compare
        diff = (layer_2 - target)
        sqdiff = (diff * diff)
        loss = sqdiff.sum(0) # mean squared error loss

        # Learn: this is the backpropagation piece
        layer_1_grad = 2*diff.dot(weights_1_2.transpose())
        weight_1_2_update = layer_1.transpose().dot(2*diff)
        weight_0_1_update = data.transpose().dot(layer_1_grad)
        
        weights_1_2 -= weight_1_2_update * alpha
        weights_0_1 -= weight_0_1_update * alpha
        print(loss[0])

@add_divider(comment='all_framework')
def all_framework(epoches, alpha, orgw_0_1, orgw_1_2):
    L1 = LinearLayer(2,3,bias=False)
    L2 = LinearLayer(3,1,bias=False)

    L1.weights.data = orgw_0_1.copy()     # comment this line to see different result
    L2.weights.data = orgw_1_2.copy()     # comment this line to see different result
    data = Tensor([[0,0],[0,1],[1,0],[1,1]], autograd=True)
    target = Tensor([[0],[1],[0],[1]], autograd=True)

    model = Sequential([L1, L2])
    sgd = SGD(parameters=model.get_parameters(), alpha=alpha)
    criterion = MSELoss()

    for i in range(epoches):
        # Empty grad
        sgd.zero_grad()

        # Predict
        pred = model.forward(data)
        
        # Compare, mean squared error loss
        loss = criterion(pred, target)

        # Learn: this is the backpropagation piece
        loss.backward()
        
        # Update weights
        sgd.step()
        
        print(loss[0])

@add_divider(comment='all_framework_crossentropy')
def all_framework_crossentropy(epoches):
    data = Tensor(np.array([1,2,1,2]), autograd=True)
    target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

    embed = EmbeddingLayer(5,3)
    model = Sequential([embed, Tanh(), LinearLayer(3,2)])
    criterion = CrossEntropyLoss()

    optim = SGD(parameters=model.get_parameters(), alpha=0.25)

    for i in range(epoches):
        # zero grad
        optim.zero_grad()
        # predict
        pred = model.forward(data)
        # loss
        loss = criterion(pred, target)
        # backward
        loss.backward()
        # step grad
        optim.step()
        # print loss
        print(loss[0])
if __name__ == '__main__':
    all_manual(epoches, alpha, orgw_0_1, orgw_1_2)
    all_framework(epoches, alpha, orgw_0_1, orgw_1_2)
    all_framework_crossentropy(epoches)
