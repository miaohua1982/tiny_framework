from .layer import Layer
from .layer import Parameter
from .tensor import Tensor

class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = self.get_name('Sigmoid_')

    def forward(self, x):
        return x.sigmoid()

    def __call__(self, x):
        return x.sigmoid()
    
    def __repr__(self):
        return '['+self.name+':()]'
    
class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = self.get_name('Tanh_')

    def forward(self, x):
        return x.tanh()

    def __call__(self, x):
        return x.tanh()
    
    def __repr__(self):
        return '['+self.name+':()]'

class Relu(Layer):
    def __init__(self):
        super(Relu, self).__init__()
        self.name = self.get_name('Relu_')

    def forward(self, x):
        return x.relu()

    def __call__(self, x):
        return x.relu()
    
    def __repr__(self):
        return '['+self.name+':()]'