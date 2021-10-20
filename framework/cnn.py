from .layer import Layer
from .parameter import Parameter
from .init import kaiming_uniform, kaiming_uniform_bias
import numpy as np


class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, bias=True, padding=0):
        super(Conv2d, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        kernel_weights = kaiming_uniform((output_channels, input_channels, kernel_size, kernel_size), a=np.sqrt(5))
        self.kernel = Parameter(self.get_name('Conv2d_Weights_'), kernel_weights, requires_grad=True)

        # add bias if needed
        if bias:
            bias_weights = kaiming_uniform_bias((output_channels, input_channels, kernel_size, kernel_size), (output_channels,))
            self.bias = Parameter(self.get_name('Conv2d_Bias_'), bias_weights, requires_grad=True)
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
        output = input.conv2d_cpp(self.input_channels, self.output_channels, self.kernel, self.bias, self.stride, self.padding)
        return output
    
    def __call__(self, input):
        return self.forward(input)

class BatchNorm1d(Layer):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_states = track_running_stats
        self.num_batches_tracked = 0
        self.mi = np.zeros(num_features, dtype=np.float32)
        self.var = np.ones(num_features, dtype=np.float32)

        self.gamma = Parameter(self.get_name('BatchNorm1d_Gamma_'), np.ones(num_features, dtype=np.float32), requires_grad=affine)
        self.beta = Parameter(self.get_name('BatchNorm1d_Betta_'), np.zeros(num_features, dtype=np.float32), requires_grad=affine)

    def forward(self, x):
        assert x.dim() == 2, "input features should have 2 dim in batchnorm1d operation"

        if self.is_training and self.track_running_states:
            self.num_batches_tracked = self.num_batches_tracked+1
        
        if self.is_training:
            if self.momentum is None:
                self.momentum = 1.0/float(self.num_batches_tracked)
            # do channel-wise normalization
            output, cur_mi, cur_var = x.batchnorm1d(self.num_features, self.gamma, self.beta, self.eps, self.affine) 
            
            self.mi = (1-self.momentum)*self.mi+self.momentum*cur_mi
            self.var = (1-self.momentum)*self.var+self.momentum*cur_var
        else:
            output = x.batchnorm1d_eval(self.mi, self.var, self.gamma, self.beta, self.eps, self.affine) 
        
        return output
    
    def __call__(self, input):
        return self.forward(input)


class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_states = track_running_stats
        self.num_batches_tracked = 0
        self.mi = np.zeros(num_features, dtype=np.float32)
        self.var = np.ones(num_features, dtype=np.float32)

        self.gamma = Parameter(self.get_name('BatchNorm2d_Gamma_'), np.ones(num_features, dtype=np.float32), requires_grad=affine)
        self.beta = Parameter(self.get_name('BatchNorm2d_Betta_'), np.zeros(num_features, dtype=np.float32), requires_grad=affine)

    def forward(self, x):
        assert x.dim() == 4, "input features should have 4 dim in batchnorm2d operation"

        if self.is_training and self.track_running_states:
            self.num_batches_tracked = self.num_batches_tracked+1
        
        if self.is_training:
            if self.momentum is None:
                self.momentum = 1.0/float(self.num_batches_tracked)
            # do channel-wise normalization
            output, cur_mi, cur_var = x.batchnorm2d(self.num_features, self.gamma, self.beta, self.eps, self.affine) 
            
            self.mi = (1-self.momentum)*self.mi+self.momentum*cur_mi
            self.var = (1-self.momentum)*self.var+self.momentum*cur_var
        else:
            output = x.batchnorm2d_eval(self.mi, self.var, self.gamma, self.beta, self.eps, self.affine) 
        
        return output
    
    def __call__(self, input):
        return self.forward(input)


        