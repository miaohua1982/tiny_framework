import numpy as np

class Optimizer(object):
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.alpha = lr
    
    def set_lr(self, lr):
        self.alpha = lr
    
    def get_lr(self):
        return self.alpha

    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for one_param in self.parameters:
            one_param.zero_grad()
class SGD(Optimizer):
    '''
    The vallina sgd optimizer
    '''
    def __init__(self, parameters, lr=0.01):
        super(SGD, self).__init__(parameters, lr)
    
    def step(self):
        for one_param in self.parameters:
            one_param.step_sgd(self.alpha)

class MoMentumSGD(Optimizer):
    '''
    The Momentum sgd optimizer
    '''
    def __init__(self, parameters, lr, momentum=0.9):
        super(MoMentumSGD, self).__init__(parameters, lr)
        self.momentum = momentum
        self.v = {}
 
        for one_param in self.parameters:
            self.v[one_param.get_name()] = np.zeros(one_param.shape)
    
    def step(self):
        for one_param in self.parameters:
            if one_param.grad is None or one_param.autograd == False:
                continue

            key = one_param.get_name()
            self.v[key] = self.v[key]*self.momentum - self.alpha*one_param.grad.data
 
            one_param.step_momentum(self.v[key])

class AdaGrad(Optimizer):
    '''
    The AdaGrad sgd optimizer
    '''
    def __init__(self, parameters, lr=0.01, eps=1e-10):
        super(AdaGrad, self).__init__(parameters, lr)
        self.eps = eps
        self.h = {}
 
        for one_param in self.parameters:
            self.h[one_param.get_name()] = np.zeros(one_param.shape)
        
    def step(self):
        for one_param in self.parameters:
            if one_param.grad is None or one_param.autograd == False:
                continue

            key = one_param.get_name()
            self.h[key] += one_param.grad.data * one_param.grad.data
            one_param.step_adagrad(self.alpha, self.h[key], self.eps)
    
class RMSprop(Optimizer):
    '''
    The RMSprop sgd optimizer
    '''
    def __init__(self, parameters, lr=0.01, decay=0.99, eps=1e-08):
        super(RMSprop, self).__init__(parameters, lr)
        self.decay = decay
        self.eps = eps
        self.h = {}
 
        for one_param in self.parameters:
            self.h[one_param.get_name()] = np.zeros(one_param.shape)

    def step(self):
        for one_param in self.parameters:
            if one_param.grad is None or one_param.autograd == False:
                continue

            key = one_param.get_name()
            self.h[key] = self.h[key]*self.decay + (1-self.decay)*one_param.grad.data * one_param.grad.data
            one_param.step_adagrad(self.alpha, self.h[key], self.eps)

class Nestrov(object):
    '''
    The Nestrov sgd optimizer
    '''
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super(Nestrov, self).__init__(parameters, lr)
        self.momentum = momentum
        self.v = {}
 
        for one_param in self.parameters:
            self.v[one_param.get_name()] = np.zeros(one_param.shape)
 
    def step(self):
        for one_param in self.parameters:
            if one_param.grad is None or one_param.autograd == False:
                continue

            key = one_param.get_name()
            self.v[key] = self.v[key]*self.momentum - self.alpha*one_param.grad.data
            one_param.step_nestrov(self.alpha, self.momentum, self.v[key])


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = {}
        self.v = {}
 
        for one_param in self.parameters:
            self.m[one_param.get_name()] = np.zeros(one_param.shape)
            self.v[one_param.get_name()] = np.zeros(one_param.shape)
        
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
