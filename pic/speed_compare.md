import torch as t
import numpy as np
import conv_operations as co


#-----------------------------------------------------------------------------
# max pool 2d speed test
max_pool2d_t = t.nn.MaxPool2d(kernel_size=2)
input_t = t.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,10],[11,12,13,14]], dtype=t.float32, requires_grad=True).reshape(1,1,4,4)
output = max_pool2d_t(input_t)

input = input_t.detach().numpy().copy()
output_max = np.zeros((1,1,2,2), dtype=np.float32)
output_max_inds =  co.maxpool2d_forward(input_t.detach().numpy(), output_max, 2, 2, 0)

# forward test
# %timeit output = max_pool2d_t(input_t)
# %timeit output_max_inds =  co.maxpool2d_forward(input, output_max, 2, 2, 0)

%%timeit 
input_t.grad = None
input_t.retain_grad()   # to enable to access grad of non-leaf node's grad
output = max_pool2d_t(input_t)
output = output.flatten()
f = output.sum()
f.backward()


print(input_t.grad)


%%timeit 
output_max = np.zeros((1,1,2,2), dtype=np.float32)
output_max_inds =  co.maxpool2d_forward(input, output_max, 2, 2, 0)
grad_output = np.ones(output_max.shape, dtype=np.float32)
input_grad = co.maxpool2d_backward(grad_output, output_max_inds, input.shape[2], input.shape[3], 2, 2, 0)

print(input_grad)


#-----------------------------------------------------------------------------
# conv2d speed test
input = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32).reshape(1,1,3,3)
input_t = t.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=t.float32, requires_grad=True).reshape(1,1,3,3)

conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
output_t = conv2d_t(input_t)
print(output_t)

output_holder = np.zeros((1, 2, 4, 4), dtype=np.float32)
padding_feat = co.conv2d_forward_withbias(input, conv2d_t.weight.data.numpy(), conv2d_t.bias.data.numpy(), output_holder, 1, 1)
print(output_holder.round(4))   # cpp library output has more precision

# forward test
# %timeit output_t = conv2d_t(input_t)
# %timeit padding_feat = co.conv2d_forward_withbias(input, conv2d_t.weight.data.numpy(), conv2d_t.bias.data.numpy(), output_holder, 1, 1)


%%timeit
input_t.grad = None
input_t.retain_grad()   # to enable to access grad of non-leaf node's grad
conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
output_t = conv2d_t(input_t)
output_t = output_t.flatten()
f = output_t.sum()
f.backward()


print(input_t.grad)


%%timeit
output_holder = np.zeros((1, 2, 4, 4), dtype=np.float32)
padding_feat = co.conv2d_forward_withbias(input, conv2d_t.weight.data.numpy(), conv2d_t.bias.data.numpy(), output_holder, 1, 1)
grad_output = np.ones((1,2,4,4), dtype=np.float32)
kernel_grad = np.zeros((2,1,2,2), dtype=np.float32)
bias_grad = np.zeros(2, dtype=np.float32)
input_grad = co.conv2d_backward_withbias(grad_output, padding_feat, conv2d_t.weight.data.numpy(), conv2d_t.bias.data.numpy(), kernel_grad, bias_grad, 1, 1)
print(input_grad)          # compare to input.grad
