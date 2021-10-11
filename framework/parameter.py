from .tensor import Tensor

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