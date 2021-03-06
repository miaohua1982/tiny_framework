from .tensor import Tensor
import numpy as np

class Parameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (np.numpy): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """

    def __init__(self, name, data=None, requires_grad=True):
        if data is None:
            data = []
        super(Parameter, self).__init__(data, autograd=requires_grad)
        
        self.name = name

    def get_name(self):
        return self.name
    
    def __repr__(self):
        return self.name+':'+super(Parameter, self).__repr__()
    
    def zero_grad(self):
        self.grad = None
       
    def step_sgd(self, alpha):
        if self.grad is None or self.autograd == False:
            return
        self.data -= self.grad.data*alpha

    def step_adam(self, alpha, m, v, eps):
        if self.grad is None or self.autograd == False:
            return
        self.data -= alpha * m / (np.sqrt(v) + eps)
    
    def step_momentum(self, v):
        if self.grad is None or self.autograd == False:
            return

        self.data += v

    def step_adagrad(self, alpha, h, eps):
        if self.grad is None or self.autograd == False:
            return

        self.data -= self.grad.data*alpha / (np.sqrt(h)+eps)

    def step_nestrov(self, alpha, momentum, v):
        if self.grad is None or self.autograd == False:
            return
        self.data += momentum*momentum*v - (1+momentum)*alpha*self.grad.data

    