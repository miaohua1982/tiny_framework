from .parameter import Parameter
from .init import kaiming_uniform_, kaiming_uniform_bias_
import numpy as np
from collections import OrderedDict, namedtuple


# add layer support
class ItemIdGen(object):
    def __init__(self):
        self.gen_id = {}
        
        self.default_items = ['Dropout2d_', 'Dropout_', 'Conv2d_', 'BatchNorm2d_', 'BatchNorm2d_Gamma_', 'BatchNorm2d_Betta_', 'Conv2d_Weights_', 'Conv2d_Bias_', 'MaxPool2d_', 'Linear_Weights_', 'Linear_Bias_', 'Sequential_', 'RNNCell_', 'LstmCell_', 'Embedding_Weights_', 'Sigmoid_', 'Tanh_', 'Relu_']
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

class Layer(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self.is_training = True
    
    def get_parameters(self):
        return list(self._parameters.values())

    def get_name(self, prefix):
        suf_id = item_id_gen.get_next_id(prefix)
        return prefix+str(suf_id)

    def eval(self):
        self.is_training = False
        for one_param in self._parameters.values():
            one_param.eval()

    def train(self):
        self.is_training = True
        for one_param in self._parameters.values():
            one_param.train()
    
    def __repr__(self):
        layer_repr = ''
        for one in self.get_parameters():
            layer_repr += '['+one.__repr__()+']\n'
        return layer_repr

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
    
    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Layer.__init__() call")
            remove_from(self.__dict__)
            #self.register_parameter(name, value)
            self._parameters[name] = value
        #elif params is not None and name in params:
        #    if value is not None:
        #        raise TypeError("cannot assign '{}' as parameter '{}' "
        #                        "(torch.nn.Parameter or None expected)"
        #                        .format(type(value), name))
        #    self.register_parameter(name, value)
        else:
            object.__setattr__(self, name, value)


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
        

class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingLayer, self).__init__()
        w = (np.random.rand(vocab_size, hidden_size)-0.5)/hidden_size
        self.embedding_weights = Parameter(self.get_name('Embedding_Weights_'), w, requires_grad=True)

    def forward(self, words):
        return self.embedding_weights.index_select(words)

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()
        self.name = self.get_name('MaxPool2d_')

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
        return '['+self.name+':()]'

# add layers supportion(layer container)
class  Sequential(Layer):
    def __init__(self, layers=None):
        super(Sequential, self).__init__()
        if layers is None:
            self._layers = OrderedDict()
        else:
            self._layers = layers
        self.name = self.get_name('Sequential_')

    def get_parameters(self):
        layers_params = []
        for one in self._layers.values():
            layers_params += one.get_parameters()
        return layers_params
    
    def eval(self):
        for one_layer in self._layers.values():
            one_layer.eval()
    
    def train(self):
        for one_layer in self._layers.values():
            one_layer.train()
    
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_layers' in self.__dict__:
            modules = self.__dict__['_layers']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
    
    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        layers = self.__dict__.get('_layers')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Layer.__init__() call")
            remove_from(self.__dict__, self._layers)
            #self.register_parameter(name, value)
            self._parameters[name] = value
        elif isinstance(value, Layer):
            if layers is None:
                raise AttributeError("cannot assign layers before Sequential.__init__() call")
            remove_from(self.__dict__, self._parameters)
            #self.register_parameter(name, value)
            self._layers[name] = value
        #elif params is not None and name in params:
        #    if value is not None:
        #        raise TypeError("cannot assign '{}' as parameter '{}' "
        #                        "(torch.nn.Parameter or None expected)"
        #                        .format(type(value), name))
        #    self.register_parameter(name, value)
        else:
            object.__setattr__(self, name, value)

    def forward(self, x):
        p = x
        for one in self._layers.values():
            p = one.forward(p)
            
        return p

    def __repr__(self):
        rep = self.name+':[\n'
        for n, layer in self._layers.items():
            rep = rep + n + ':[\n' + layer.__repr__()+']\n'
        return rep
    
