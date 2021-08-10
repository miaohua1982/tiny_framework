class SGD(object):
    '''
    The vallina sgd optimizer
    '''
    def __init__(self, parameters, alpha):
        self.parameters = parameters
        self.alpha = alpha
    
    def zero_grad(self):
        for one_param in self.parameters:
            one_param.zero_grad()
    
    def step(self):
        for one_param in self.parameters:
            one_param.step(self.alpha)