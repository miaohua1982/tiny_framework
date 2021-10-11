from .layer import Layer

class Dropout(Layer):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.name = self.get_name('Dropout_')

        assert p >= 0 and p < 1, "The drop prob must be in [0, 1)"
        self.prob = p
        
    def forward(self, x):
        return x.dropout(self.prob, self.is_training)
    
    def __call__(self, x):
        self.forward(x)
    
    def __repr__(self):
        return '['+self.name+':Tensor()]'

class Dropout2d(Layer):
    def __init__(self, p=0.5):
        super(Dropout2d, self).__init__()
        self.name = self.get_name('Dropout2d_')

        assert p >= 0 and p < 1, "The drop prob must be in [0, 1)"
        self.prob = p
        
    def forward(self, x):
        return x.dropout2d(self.prob, self.is_training)
    
    def __call__(self, x):
        self.forward(x)
    
    def __repr__(self):
        return '['+self.name+':Tensor()]'
    
