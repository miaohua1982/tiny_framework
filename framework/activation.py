from .layer import Layer
from .layer import Parameter
from .tensor import Tensor

class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.parameters.append(Parameter(self.get_name('Sigmoid_'), Tensor([])))

    def forward(self, x):
        return x.sigmoid()

    def __call__(self, x):
        return x.sigmoid()
    
    def __repr__(self):
        return '['+self.name+'Tensor()]'
    
class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()
        self.parameters.append(Parameter(self.get_name('Tanh_'), Tensor([])))

    def forward(self, x):
        return x.tanh()

    def __call__(self, x):
        return x.tanh()
    
    def __repr__(self):
        return '['+self.name+'Tensor()]'
