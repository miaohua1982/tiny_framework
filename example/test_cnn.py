import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep)

from framework.cnn import Conv2d
from framework.layer import MaxPool2d
from framework.tensor import Tensor
import torch as t
import numpy as np


input = Tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]).reshape(1,1,3,3), autograd=True)
kernel = np.array([[1,2],[2,1]]).reshape(1,1,2,2)
kernel = np.concatenate([kernel, kernel])
bias = np.array([2,2])
input_t = t.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=t.float32, requires_grad=True).reshape(1,1,3,3)
kernel_t = t.tensor(kernel.tolist(), dtype=t.float32, requires_grad=True).reshape(2,1,2,2)
bias_t = t.tensor([2,2], dtype=t.float32, requires_grad=True)

print(input.shape)
print(input)
print(kernel.shape)
print(kernel)
print(bias.shape)
print(bias)

#-------------------------------------------------------------------------
# conv2d test
conv2d = Conv2d(input_channels=1, output_channels=2, kernel_size=2, padding=1)
conv2d.set_weight(kernel)
conv2d.set_bias(bias)
print(conv2d)
print(conv2d.kernel)
print(conv2d.bias)

conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, padding=1)
conv2d_t.weight.data = kernel_t
conv2d_t.bias.data = bias_t

input_t.retain_grad()   # to enable to access grad of non-leaf node's grad
output_t = conv2d_t(input_t)
print(output_t.shape)
print(output_t)
output_t = output_t.flatten()
f = output_t.sum()
f.backward()
print('kernel_t grad:')
print(conv2d_t.weight.grad)  # compare to conv2d.kernel.grad
print('input_t grad:')
print(input_t.grad)          # compare to input.grad
print('bias grad:')
print(conv2d_t.bias.grad)

output = conv2d(input)
output = output.flatten()
f = output.sum()
print(output.shape)
print(output)
print(f)
f.backward()
print('kernel grad:')
print(conv2d.kernel.grad)
print('input grad:')
print(input.grad)
print('bias grad:')
print(conv2d.bias.grad)

#--------------------------------------------------------------------
#max pool 2d test
max_pool2d = MaxPool2d(kernel_size=2)
max_pool2d_t = t.nn.MaxPool2d(kernel_size=2)
input = Tensor(np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10],[11,12,13,14]]).reshape(1,1,4,4), autograd=True)
input_t = t.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,10],[11,12,13,14]], dtype=t.float32, requires_grad=True).reshape(1,1,4,4)

output = max_pool2d(input)
print(output)
output = output.flatten()
f = output.sum()
print(f)
f.backward()
print('input grad:')
print(input.grad)

input_t.retain_grad()   # to enable to access grad of non-leaf node's grad
output_t = max_pool2d_t(input_t)
print(output_t)
output_t = output_t.flatten()
f_t = output_t.sum()
print(f_t)
f_t.backward()
print('input grad:')
print(input_t.grad)