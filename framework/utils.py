
import numpy as np
from .tensor import Tensor


def save_model(model, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    import pickle
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def all(tensor):
    return tensor.flatten().sum().item() == tensor.el_num()

def concatenate(cons, dim=0):
    new = np.concatenate(cons, dim)
    if cons[0].autograd:
        new_tensor = Tensor(new, autograd=True, creator=cons, create_op='concatenate')
        new_tensor.dim = dim
        return new_tensor
    
    return Tensor(new)

def relu(input):
    return input.relu()