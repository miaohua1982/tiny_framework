import torch as t
import numpy as np
import conv_operations as co


# test the test
assert co.add(1, 2) == 3

# compare conv2d with no padding
input = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32).reshape(1,1,3,3)
kernel = np.array([[1,2],[2,1]], dtype=np.float32).reshape(1,1,2,2)
kernel = np.concatenate([kernel, kernel])
bias = np.array([2,2], dtype=np.float32)

input_t = t.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=t.float32, requires_grad=True).reshape(1,1,3,3)
kernel_t = t.tensor(kernel.tolist(), dtype=t.float32, requires_grad=True).reshape(2,1,2,2)
bias_t = t.tensor([2,2], dtype=t.float32, requires_grad=True)

print(input.shape, input.dtype)
print(input)
print(kernel.shape, kernel.dtype)
print(kernel)
print(bias.shape, bias.dtype)
print(bias)

#-------------------------------------------------------------------------
# regular conv2d test without padding
print('----------------------------reqular conv2d test without padding--------------------------')
conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
conv2d_t.weight.data = kernel_t
conv2d_t.bias.data = bias_t
output_t = conv2d_t(input_t)
print(output_t.shape)
print(output_t)

output_holder = np.zeros((1, 2, 2, 2), dtype=np.float32)
padding_feat = np.zeros(input.shape, dtype=np.float32)
co.conv2d_forward_withbias(input, kernel, bias, padding_feat, output_holder, 1, 0)
print(output_holder)
print('Two are same:',np.all(output_t.detach().numpy() == output_holder))

# radnom conv2d test without padding
print('----------------------------random conv2d test without padding--------------------------')
conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
output_t = conv2d_t(input_t)
print(output_t.shape)
print(output_t)

output_holder = np.zeros((1, 2, 2, 2), dtype=np.float32)
padding_feat = np.zeros(input.shape, dtype=np.float32)
co.conv2d_forward_withbias(input, conv2d_t.weight.data.numpy(), conv2d_t.bias.data.numpy(), padding_feat, output_holder, 1, 0)
print(output_holder.shape)
print(output_holder.round(4))   # cpp library output has more precision
print('Two are same:',np.all(output_t.detach().numpy().round(4) == output_holder.round(4))) # compare under same precision

# regular conv2d test with padding
print('----------------------------regular conv2d test with padding--------------------------')
conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, padding=1)
conv2d_t.weight.data = kernel_t
conv2d_t.bias.data = bias_t
output_t = conv2d_t(input_t)
print(output_t.shape)
print(output_t)

output_holder = np.zeros((1, 2, 4, 4), dtype=np.float32)
padding_feat = np.zeros((1, 1, 5, 5), dtype=np.float32)   # height+2*p, width+2*p
co.conv2d_forward_withbias(input, kernel, bias, padding_feat, output_holder, 1, 1)
print(output_holder.shape)
print(output_holder)
print('Two are same:',np.all(output_t.detach().numpy() == output_holder))

# radnom conv2d test with padding
print('----------------------------random conv2d test with padding--------------------------')
conv2d_t = t.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
output_t = conv2d_t(input_t)
print(output_t.shape)
print(output_t)

output_holder = np.zeros((1, 2, 4, 4), dtype=np.float32)
padding_feat = np.zeros((1, 1, 5, 5), dtype=np.float32)
co.conv2d_forward_withbias(input, conv2d_t.weight.data.numpy(), conv2d_t.bias.data.numpy(), padding_feat, output_holder, 1, 1)
print(output_holder.shape)
print(output_holder.round(4))   # cpp library output has more precision
print('Two are same:',np.all(output_t.detach().numpy().round(4) == output_holder.round(4))) # compare under same precision

#----------------------------------------------------------------------------------------------------
# random conv2d backward test with padding
print('----------------------------random conv2d backward test with padding--------------------------')
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
print('kernel_t grad:', conv2d_t.weight.grad.shape)
print(conv2d_t.weight.grad)  # compare to conv2d.kernel.grad
print('input_t grad:', input_t.shape)
print(input_t.grad)          # compare to input.grad
print('bias grad:', conv2d_t.bias.grad.shape)
print(conv2d_t.bias.grad)

output_holder = np.zeros((1, 2, 4, 4), dtype=np.float32)
padding_feat = np.zeros((1, 1, 5, 5), dtype=np.float32)   # height+2*p, width+2*p
co.conv2d_forward_withbias(input, kernel, bias, padding_feat, output_holder, 1, 1)
print(output_holder.shape)
print(output_holder)
print(padding_feat)

grad_output = np.ones((1,2,4,4), dtype=np.float32)
#input_grad = np.zeros((1,1,5,5), dtype=np.float32)
kernel_grad = np.zeros((2,1,2,2), dtype=np.float32)
bias_grad = np.zeros(2, dtype=np.float32)

input_grad = co.conv2d_backward(grad_output, padding_feat, kernel, bias, kernel_grad, bias_grad, 1, 1)
print('kernel_t grad:')
print(kernel_grad)         # compare to conv2d.kernel.grad
print('input_t grad:', input_grad.shape)
print(input_grad)          # compare to input.grad
print('bias grad:')
print(bias_grad)

