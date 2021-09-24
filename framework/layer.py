from .tensor import Tensor
from .init import kaiming_uniform_, kaiming_uniform_bias_
import numpy as np


# add layer support
class ItemIdGen(object):
    def __init__(self):
        self.gen_id = {}
        
        self.default_items = ['Conv2d_', 'BatchNorm2d_', 'BatchNorm2d_Gamma_', 'BatchNorm2d_Betta_', 'Conv2d_Weights_', 'Conv2d_Bias_', 'MaxPool2d_', 'Linear_Weights_', 'Linear_Bias_', 'Sequential_', 'RNNCell_', 'LstmCell_', 'Embedding_Weights_', 'Sigmoid_', 'Tanh_', 'Relu_']
        for one_item in self.default_items:
            self.gen_id[one_item] = 0
    
    def add_item(self, item):
        self.default_items.append(item)
        self.gen_id[item] = 0
    
    def get_next_id(self, item):
        assert item in self.gen_id, 'The item %s is not in Id Gen base' % item
        
        cur_id = self.gen_id[item]
        self.gen_id[item] += 1
        
        return cur_id

item_id_gen = ItemIdGen()

class Parameter(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        
    def get_name(self):
        return self.name
    
    def get_value(self):
        return self.value

    def shape(self):
        return self.value.shape
    
    def step(self, alpha):
        self.value.step(alpha)
        
    def zero_grad(self):
        self.value.zero_grad()
    
    def eval(self):
        self.value.eval()

    def train(self):
        self.value.train()

    def __repr__(self):
        return self.name+':'+self.value.__repr__()

class Layer(object):
    def __init__(self):
        self.parameters = []
        self.is_training = True
    
    def get_parameters(self):
        return self.parameters

    def get_name(self, prefix):
        suf_id = item_id_gen.get_next_id(prefix)
        return prefix+str(suf_id)

    def eval(self):
        self.is_training = False
        for one_param in self.parameters:
            one_param.eval()

    def train(self):
        self.is_training = True
        for one_param in self.parameters:
            one_param.train()
    
    def __repr__(self):
        layer_repr = ''
        for one in self.get_parameters():
            layer_repr += '['+one.__repr__()+']\n'
        return layer_repr

class LinearLayer(Layer):
    def __init__(self, inns, outs, bias=True):
        super(LinearLayer, self).__init__()
        
        self.weights = Tensor(np.random.rand(inns, outs)*np.sqrt(2.0/inns), autograd=True)
        self.parameters.append(Parameter(self.get_name('Linear_Weights_'), self.weights))
        
        if bias:
            self.bias = Tensor(np.zeros(outs), autograd=True)
            self.parameters.append(Parameter(self.get_name('Linear_Bias_'), self.bias))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=np.sqrt(5))
        
        if self.bias is not None:
            kaiming_uniform_bias_(self.weights.shape, self.bias)


    def forward(self, x):
        y = x.mm(self.weights)
        if self.bias is not None:
            y += self.bias.expand(0, x.shape[0])
        
        return y

    def __call__(self, x):
        return self.forward(x)
        
class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingLayer, self).__init__()
        w = (np.random.rand(vocab_size, hidden_size)-0.5)/hidden_size
        self.embedding_weights = Tensor(w, autograd=True)
        self.parameters.append(Parameter(self.get_name('Embedding_Weights_'), self.embedding_weights))

    def forward(self, words):
        return self.embedding_weights.index_select(words)

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()
        self.parameters.append(Parameter(self.get_name('MaxPool2d_'), Tensor([])))
        if stride is None:
            stride = kernel_size
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        return input.max_pool2d_cpp(self.kernel_size, self.stride, self.padding)
    
    def __call__(self, input):
        return self.forward(input)

    def __repr__(self):
        return self.name+':[\n'+super().__repr__()+'\n]\n'

# add layers supportion(layer container)
class  Sequential(Layer):
    def __init__(self, layers=None):
        super(Sequential, self).__init__()
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.name = self.get_name('Sequential_')

    def get_parameters(self):
        layers_params = []
        for one in self.layers:
            layers_params += one.get_parameters()
        return layers_params
    
    def eval(self):
        for one_layer in self.layers:
            one_layer.eval()
    
    def train(self):
        for one_layer in self.layers:
            one_layer.train()
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        p = x
        for one in self.layers:
            p = one.forward(p)
            
        return p

    def __repr__(self):
        return self.name+':[\n'+super().__repr__()+']\n'
    
