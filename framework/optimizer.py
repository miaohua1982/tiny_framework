class SGD(object):
    '''
    The vallina sgd optimizer
    '''
    def __init__(self, parameters, alpha, decay=1.0):
        self.parameters = parameters
        self.alpha = alpha
        self.decay = decay
    
    def decay_lr(self):
        self.alpha *= self.decay
        
    def zero_grad(self):
        for one_param in self.parameters:
            one_param.zero_grad()
    
    def step(self):
        for one_param in self.parameters:
            one_param.step(self.alpha)