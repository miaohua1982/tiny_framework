import numpy as np

class SGD(object):
    '''
    The vallina sgd optimizer
    '''
    def __init__(self, parameters, lr=0.01, decay=1.0):
        self.parameters = parameters
        self.alpha = lr
        self.decay = decay
    
    def decay_lr(self):
        self.alpha *= self.decay
        
    def zero_grad(self):
        for one_param in self.parameters:
            one_param.zero_grad()
    
    def step(self):
        for one_param in self.parameters:
            one_param.step_sgd(self.alpha)


class MoMentumSGD(object):
    '''
    The vallina sgd optimizer
    '''
    def __init__(self, parameters, lr, momentum=0.9):
        self.parameters = parameters
        self.alpha = lr
        self.momentum = momentum
        self.v = {}
 
        for one_param in self.parameters:
            self.v[one_param.get_name()] = np.zeros(one_param.shape)
        
    def zero_grad(self):
        for one_param in self.parameters:
            one_param.zero_grad()
    
    def step(self):
        for one_param in self.parameters:
            if one_param.grad is None or one_param.autograd == False:
                continue

            key = one_param.get_name()
            self.v[key] = self.v[key]*self.momentum - self.alpha*one_param.grad.data
 
            one_param.step_momentum(self.v[key])

class Adam(object):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = {}
        self.v = {}
 
        for one_param in self.parameters:
            self.m[one_param.get_name()] = np.zeros(one_param.shape)
            self.v[one_param.get_name()] = np.zeros(one_param.shape)
        
    def zero_grad(self):
        for one_param in self.parameters:
            one_param.zero_grad()
    
    def step(self):
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
 
        for one_param in self.parameters:
            if one_param.grad is None or one_param.autograd == False:
                continue
            key = one_param.get_name()
            self.m[key] += (1 - self.beta1) * (one_param.grad.data - self.m[key])
            self.v[key] += (1 - self.beta2) * (one_param.grad.data**2 - self.v[key])
 
            one_param.step_adam(lr_t, self.m[key], self.v[key], self.eps)
