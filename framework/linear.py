import numpy as np
from .layer import Layer
from .parameter import Parameter
from .init import kaiming_uniform_, kaiming_uniform_bias_

class LinearLayer(Layer):
    def __init__(self, inns, outs, bias=True):
        super(LinearLayer, self).__init__()
        
        self.weights = Parameter(self.get_name('Linear_Weights_'), np.random.rand(inns, outs)*np.sqrt(2.0/inns), requires_grad=True)
        
        if bias:
            self.bias = Parameter(self.get_name('Linear_Bias_'), np.zeros(outs), requires_grad=True)
        else:
            self.bias = None
        
        self.inns = inns
        self.outs = outs

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=np.sqrt(5))
        
        if self.bias is not None:
            kaiming_uniform_bias_(self.weights.shape, self.bias)

    def get_input_features(self):
        return self.inns
    
    def get_output_features(self):
        return self.outs
    
    def forward(self, x):
        y = x.mm(self.weights)
        if self.bias is not None:
            y += self.bias.expand(0, x.shape[0])
        
        return y

    def __call__(self, x):
        return self.forward(x)
