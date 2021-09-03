import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+os.path.sep+os.path.pardir+os.path.sep)

from framework.cnn import Conv2d
from framework.tensor import Tensor
import torch as t
import numpy as np


input = Tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]).reshape(1,1,3,3), autograd=True)
kernel = np.array([[1,2],[2,1]]).reshape(1,1,2,2)
kernel = np.concatenate([kernel, kernel])
input_t = t.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=t.float32, requires_grad=True).reshape(1,1,3,3)
kernel_t = t.tensor(kernel.tolist(), dtype=t.float32, requires_grad=True).reshape(2,1,2,2)

print(input.shape)
print(input)
print(kernel.shape)
print(kernel)

conv2d = Conv2d(input_channels=1, output_channels=2, kernel_size=2)
conv2d.set_weight(kernel)
print(conv2d)
print(conv2d.kernel)

conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
conv2d_t.weight.data = kernel_t
conv2d_t.bias.data = t.tensor([0,0], dtype=t.float32, requires_grad=True)

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