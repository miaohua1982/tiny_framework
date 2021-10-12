from .layer import Layer

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
