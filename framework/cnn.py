from .layer import Layer, Parameter
from .tensor import Tensor
from .init import kaiming_uniform, kaiming_uniform_bias
import numpy as np


class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, bias=True, padding=0):
        super(Conv2d, self).__init__()
        self.name = self.get_name('Conv2d_')

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        kernel_weights = kaiming_uniform((output_channels, input_channels, kernel_size, kernel_size), a=np.sqrt(5))
        self.kernel = Tensor(kernel_weights, autograd=True)
        self.parameters.append(Parameter(self.get_name('Conv2d_Weights_'), self.kernel))

        # add bias if needed
        if bias:
            bias_weights = kaiming_uniform_bias((output_channels, input_channels, kernel_size, kernel_size), (output_channels,))
            self.bias = Tensor(bias_weights, autograd=True)
            self.parameters.append(Parameter(self.get_name('Conv2d_Bias_'), self.bias))
        else:
            self.bias = None
   
    def set_weight(self, kernel):
        assert kernel.ndim == 4, "the shape of kernel must be 4 in conv2d"
        assert kernel.shape[0] == self.output_channels, "first dim size must be equal to output_channels"
        assert kernel.shape[1] == self.input_channels, "second dim size must be equal to input_channels"
        assert kernel.shape[2] == self.kernel_size and kernel.shape[3] == self.kernel_size, "the width and height must be equal to kernel size"
        assert self.kernel.grad is None, "you can not change parameter during backward"

        np.copyto(self.kernel.data, kernel)

    def set_bias(self, bias):
        assert bias.ndim == 1, "the shape of bias must be 1 in conv2d"
        assert bias.shape[0] == self.output_channels, "length of bias must be equal to output_channels"
        assert self.bias.grad is None, "you can not change parameter during backward"

        np.copyto(self.bias.data, bias)

    def forward(self, input):
        output = input.conv2d(self.input_channels, self.output_channels, self.kernel, self.bias, self.stride, self.padding)
        return output
    
    def __call__(self, input):
        return self.forward(input)

    def __repr__(self):
        return self.name+':[\n'+super().__repr__()+'\n]\n'