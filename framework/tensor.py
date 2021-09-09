import numpy as np

class Tensor(object):
    def __init__(self, data, autograd=False, creator=None, create_op=None, id=None):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.creator = creator
        self.create_op = create_op
        self.autograd = autograd
        self.grad = None
        self.is_eval = False
        self.children = {}
        self.restore_children = {}
        if id is None:
            id = np.random.choice(10000)
        self.id = id
        
        if creator is not None:
            for c in creator:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
                # save the counter sync
                if self.id not in c.restore_children:
                    c.restore_children[self.id] = 1
                else:
                    c.restore_children[self.id] += 1

                # for rnn
                c.restore_graph() # just flow back it if needed,                    

    def restore_graph(self):
        '''
        if we want to use a middle tensor(which has been grad flowed, and up to its parents) **again**,
        use the function restore graph, it restores the children flow counter up to its parents, and 
        it **clears** its grad to prevent from propgating the previous operation grad again
        '''
        if self.creator is not None and self.grad is not None: # it is a middle variable, and has ever been grad flowed
            for c in self.creator:
                assert self.id in c.children, 'we are in restore graph, so its id should be in creator children list'
                c.children[self.id] = c.restore_children[self.id]
                self.grad = None
                # just flow back it if needed, if its parents is not root, still have creators, then restore back through
                c.restore_graph()
                
            
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data+other.data, autograd=True, creator=(self, other), create_op='add')
        return Tensor(self.data+other.data)
        
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data*-1, autograd=True, creator=(self,), create_op='neg')
        return Tensor(self.data*-1)
    
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data-other.data, autograd=True, creator=(self, other), create_op='sub')
        return Tensor(self.data-other.data)
        
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data*other.data, autograd=True, creator=(self, other), create_op='mul')
        return Tensor(self.data*other.data)
    
    def __eq__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data==other.data, autograd=True, creator=(self, other), create_op='eq')
        return Tensor(self.data==other.data)

    def sum(self, dim=0):
        assert self.data.ndim>dim, 'axis %d is out of bounds for array of dimension %d' % (dim, self.data.ndim)
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, creator=(self,), create_op='sum_'+str(dim))
        return Tensor(self.data.sum(dim))
    
    def mean(self, dim=0):
        assert self.data.ndim>dim, 'axis %d is out of bounds for array of dimension %d' % (dim, self.data.ndim)
        if self.autograd:
            return Tensor(self.data.mean(dim), autograd=True, creator=(self,), create_op='mean_'+str(dim))
        return Tensor(self.data.mean(dim))
    
    def expand(self, dim, copies):
        assert self.data.ndim>=dim, 'axis %d is out of bounds for array of dimension %d' % (dim, self.data.ndim)
        if self.autograd:
            return Tensor(np.expand_dims(self.data, axis=dim).repeat(copies, axis=dim), autograd=True, creator=(self,), create_op='expand_'+str(dim))
        return Tensor(np.expand_dims(self.data, axis=dim).repeat(copies, axis=dim))
    
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, creator=(self,), create_op='transpose')
        return Tensor(self.data.transpose())
    
    def mm(self, other):
        if self.autograd:
            return Tensor(self.data.dot(other.data), autograd=True, creator=(self, other), create_op='mm')
        return Tensor(self.data.dot(other.data))
    
    def view(self, new_shape):
        if self.autograd:
            new = Tensor(self.data.reshape(new_shape), autograd=True, creator=(self,), create_op='view')
            new.org_shape = self.data.shape
            return new
        return Tensor(self.data.reshape(new_shape))

    def relu(self):
        if self.autograd:
            return Tensor(np.where(self.data>0, self.data, 0), autograd=True, creator=(self,), create_op='relu')
        return Tensor(np.where(self.data>0, self.data, 0))
    
    def sigmoid(self):
        new = 1/(1+np.exp(-self.data))
        if self.autograd:
            return Tensor(new, autograd=True, creator=(self,), create_op='sigmoid')
        return Tensor(new)
    
    def max_pool2d(self, kernel_size, stride, padding):
        assert self.data.ndim == 4, "the shape of input data must be 4 in max_pool2d"
        assert self.data.shape[2] >= kernel_size and  self.data.shape[3] >= kernel_size, "the width and height of input data must be greater or equal than kernel size"

        bs, inns, h, w = self.data.shape
        dh = (h-kernel_size)//stride+1
        dw = (w-kernel_size)//stride+1

        output_max = np.zeros((bs, inns, dh, dw))
        output_max_inds = np.zeros((bs, inns, dh, dw*2), dtype=np.int)
        for b in range(bs):
            for ins in range(inns):
                for i in range(0, h-kernel_size+1, stride):
                    for j in range(0, w-kernel_size+1, stride):
                        # output_max[b,:,i//stride,j//stride] = np.max(self.data[b,:,i:i+kernel_size,j:j+kernel_size], axis=(1,2))
                        pos_h = i//stride
                        pos_w = j//stride
                        max_val = -10000000
                        max_pos_h = -1
                        max_pos_w = -1
                        for kh in range(i, i+kernel_size):
                            for kw in range(j, j+kernel_size):
                                if max_val < self.data[b, ins, kh, kw]:
                                    max_val = self.data[b, ins, kh, kw]
                                    max_pos_h = kh
                                    max_pos_w = kw
                        output_max[b,ins,pos_h,pos_w] = max_val
                        output_max_inds[b,ins,pos_h,pos_w*2] = max_pos_h
                        output_max_inds[b,ins,pos_h,pos_w*2+1] = max_pos_w
        
        if self.autograd:
            new = Tensor(output_max, autograd=True, creator=(self,), create_op='max_pool2d')
            new.output_max_inds = output_max_inds
            new.org_shape = self.data.shape
            return new
        return Tensor(output_max)
    
    def conv2d_subarea(self, ind_h, ind_w, kernel_size, padding):
        
        w = kernel_size
        if ind_w >= padding:
            sw = ind_w-padding
            pos_ws = 0
        else:
            sw = 0
            w = kernel_size-(padding-ind_w)
            pos_ws = padding-ind_w
        
        h = kernel_size
        if ind_h >= padding:
            sh = ind_h-padding
            pos_hs = 0
        else:
            sh = 0
            h = kernel_size-(padding-ind_h)
            pos_hs = padding-ind_h
        
    def conv2d(self, input_channels, output_channels, kernel, bias, stride=1, padding=0):
        assert kernel.data.ndim == 4, "the shape of kernel must be 4 in conv2d"
        assert kernel.data.shape[0] == output_channels, "first dim size must be equal to output_channels"
        assert kernel.data.shape[1] == input_channels, "second dim size must be equal to input_channels"
        assert self.data.ndim == 4, "the shape of input data must be 4 in conv2d"
        assert self.data.shape[2] >= kernel.data.shape[2] and self.data.shape[3] >= kernel.data.shape[3], "the shape of input data must be greater than kernel shape"
        assert stride >= 1, "stride must be greater or equal than 1"
        assert padding >= 0, "padding must be greater or equal than 0"

        _, _, kh, kw = kernel.shape
        db, inns, dh, dw = self.shape
        output_width = (dw+padding*2-kw)//stride+1
        output_height = (dh+padding*2-kh)//stride+1

        input_data = self.data
        if padding > 0:
            input_data = np.zeros((db, inns, dh+padding*2, dw+padding*2))
            input_data[:,:,padding:padding+dh, padding:padding+dw] = self.data
        
        output = np.zeros((db, output_channels, output_height, output_width))
        for b in range(db):
            for out in range(output_channels):
                for i in range(0, dh+padding*2-kh+1, stride):
                    for j in range(0, dw+padding*2-kw+1, stride):
                        input = input_data[b, :, i:i+kh, j:j+kw]
                        s = input*kernel.data[out]
                        output[b, out, i//stride, j//stride] = s.sum()
        if bias is not None:
            bias_r = bias.data.reshape(output_channels, 1, 1).repeat(output_height, axis=1).repeat(output_width, axis=2)
            output += bias_r
        
        if self.autograd:
            if bias is not None:
                new = Tensor(output, autograd=True, creator=(self, kernel, bias), create_op='conv2d')
                new.has_bias = True
            else:
                new = Tensor(output, autograd=True, creator=(self, kernel), create_op='conv2d')
                new.has_bias = False
            new.stride = stride
            new.padding = padding
            
            if padding > 0 :
                new.padding_input_data = input_data
            return new
        
        return Tensor(output)

    def flatten(self):
        new_data = self.data.flatten()
        if self.autograd:
            new = Tensor(new_data, autograd=True, creator=(self,), create_op='flatten')
            new.org_shape = self.data.shape
            return new
        
        return Tensor(new_data)


    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True, creator=(self,), create_op='tanh')
        return Tensor(np.tanh(self.data))
    
    def index_select(self, inds):
        if self.autograd:
            new = Tensor(self.data[inds.data], autograd=True, creator=(self,), create_op='index_select')
            new.ind_sel = inds
            return new
        return Tensor(self.data[inds.data])
    
    def cross_entropy(self, gt):
        '''
        gt is assumed to have the shape (n,1) or (n,), each element is in [0,n_classes-1] (n is the number of samples)
        self.data is assumed to have shape (n, n_classes)
        '''
        assert self.data.ndim <= 2, 'the data''s dims should be smaller than 2'
        assert gt.data.ndim <= 2, 'the target inds array''s ndim should be equal to 1'
        
        softmax_output = np.exp(self.data)/np.exp(self.data).sum(axis=self.data.ndim-1, keepdims=True)
        gt_inds = gt.data.flatten()
        softmax_output = softmax_output.reshape(len(gt_inds), -1)
        loss = -np.log(softmax_output)[np.arange(len(gt_inds)), gt_inds]
        loss = loss.mean()     

        if self.autograd:
            new = Tensor([loss], autograd=True, creator=(self,), create_op='cross_entropy')
            new.gt = gt_inds
            new.softmax_output = softmax_output
            return new
        return Tensor([loss])
        
    def check_creator_grad_count(self):
        for c in self.children:
            if self.children[c] != 0:
                return False
        return True
    
    def backward(self, grad=None, child_grad_node=None):
        assert self.is_eval == False, "model is in eval mode, can not do backward"
        if not self.autograd:
            return
        
        if child_grad_node is not None:
            if self.children[child_grad_node.id] == 0:
                assert self.children[child_grad_node.id] != 0, \
                'creator %d'' children %d has grad count == 0, backprop can has one pass' % (self.id, child_grad_node.id)
            else:
                self.children[child_grad_node.id] -= 1

        if grad is None:
            grad = Tensor(np.ones_like(self.data, dtype=np.float, shape=self.shape))
            
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
   
        if self.creator is not None and self.check_creator_grad_count():
            if self.create_op == 'add':
                self.creator[0].backward(self.grad, self)
                self.creator[1].backward(self.grad, self)
            elif self.create_op == 'neg':
                self.creator[0].backward(self.grad.__neg__(), self)
            elif self.create_op == 'sub':
                self.creator[0].backward(self.grad, self)
                self.creator[1].backward(self.grad.__neg__(), self)
            elif self.create_op == 'mul':
                self.creator[0].backward(self.grad*self.creator[1], self)
                self.creator[1].backward(self.grad*self.creator[0], self)
            elif self.create_op == 'transpose':
                self.creator[0].backward(self.grad.transpose(), self)
            elif self.create_op == 'mm':
                c0 = Tensor(self.creator[0].data, autograd=False)  # no auto grad
                c1 = Tensor(self.creator[1].data, autograd=False)  # no auto grad
                self.creator[0].backward(self.grad.mm(c1.transpose()), self)
                self.creator[1].backward(c0.transpose().mm(self.grad), self)
            elif self.create_op[:4] == 'sum_':
                dim = int(self.create_op[4:])
                new = self.grad.expand(dim, self.creator[0].data.shape[dim])
                self.creator[0].backward(new, self)
            elif self.create_op[:7] == 'expand_':
                dim = int(self.create_op[7:])
                new = self.grad.sum(dim)
                self.creator[0].backward(new, self)
            elif self.create_op == 'relu':
                factor = np.where(self.data>0, 1, 0)
                self.creator[0].backward(Tensor(self.grad.data*factor), self)
            elif self.create_op == 'sigmoid':
                new = Tensor(self.data*(1-self.data))*self.grad
                self.creator[0].backward(new, self)
            elif self.create_op == 'tanh':
                new = Tensor(1-self.data*self.data)*self.grad
                self.creator[0].backward(new, self)
            elif self.create_op == 'index_select':
                new_grad = np.zeros_like(self.creator[0].data)
                inds = self.ind_sel.data.flatten()
                grad_ = self.grad.data.reshape(len(inds),-1)
                for i in range(len(inds)):
                    new_grad[inds[i]] += grad_[i]
                self.creator[0].backward(Tensor(new_grad), self)
            elif self.create_op == 'cross_entropy':
                new_grad = self.softmax_output.copy()
                new_grad[np.arange(len(self.gt)), self.gt] -= 1
                new_grad = Tensor(new_grad)*self.grad
                self.creator[0].backward(new_grad, self)
            elif self.create_op[:5] == 'mean_':
                dim = int(self.create_op[5:])
                copies = self.creator[0].data.shape[dim]
                new = self.grad.expand(dim, copies)
                new.data = new.data*1.0/copies
                self.creator[0].backward(new, self)
            elif self.create_op == 'flatten':
                new_grad = self.grad.data.reshape(self.org_shape)
                self.creator[0].backward(Tensor(new_grad), self)
            elif self.create_op == 'view':
                new_grad = self.grad.data.reshape(self.org_shape)
                self.creator[0].backward(Tensor(new_grad), self)
            elif self.create_op == 'eq':
                mask = (self.cretor[0].data == self.creator[1].data).float()
                self.creator[0].backward(Tensor(self.grad.data*mask), self)
                self.creator[1].backward(Tensor(self.grad.data*mask), self)
            elif self.create_op == 'max_pool2d':
                new_grad = np.zeros(self.org_shape)
                bs, inns, _, _ = self.output_max_inds.shape
                _, _, gh, gw = self.grad.shape
                for b in range(bs):
                    for ins in range(inns):
                        for i in range(gh):
                            for j in range(gw):
                                pos_h = self.output_max_inds[b, ins, i, j*2]
                                pos_w = self.output_max_inds[b, ins, i, j*2+1]
                                new_grad[b, ins, pos_h, pos_w] = self.grad.data[b, ins, i, j]
                self.creator[0].backward(Tensor(new_grad), self)

            elif self.create_op == 'conv2d':
                bs, input_channels, dh, dw = self.creator[0].shape
                output_channels, ki, kh, kw = self.creator[1].shape
                
                # grad for bias
                if self.has_bias:
                    grad_bias = self.grad.data.sum(axis=(0,2,3))
                    self.creator[2].backward(Tensor(grad_bias), self)

                # grad for kernel
                input_data = self.creator[0].data
                padding = self.padding
                if padding>0:
                    input_data = self.padding_input_data
                grad_kernel = np.zeros((output_channels, input_channels, kh, kw))
                for b in range(bs):
                    for out in range(output_channels):
                        for i in range(0, dh+padding*2-kh+1, self.stride):
                            for j in range(0, dw+padding*2-kw+1, self.stride):
                                input = input_data[b, :, i:i+kh, j:j+kw]  # input_channels*kh*kw
                                grad_kernel[out] += input*self.grad.data[b, out, i//self.stride, j//self.stride]
                if padding>0:
                    del self.padding_input_data
                self.creator[1].backward(Tensor(grad_kernel), self)

                # grad for input data
                kernel = self.creator[1].data
                grad_input = np.zeros((bs, input_channels, dh+padding*2, dw+padding*2))
                for b in range(bs):
                    for out in range(output_channels):
                        for i in range(0, dh+padding*2-kh+1, self.stride):
                            for j in range(0, dw+padding*2-kw+1, self.stride):
                                grad_input[b, :, i:i+kh, j:j+kw] += kernel[out]*self.grad.data[b, out, i//self.stride, j//self.stride]
                self.creator[0].backward(Tensor(grad_input[:,:,padding:dh+padding,padding:dw+padding]), self)
      
            
    def zero_grad(self):
        self.grad = None
       
    def step(self, alpha):
        if self.grad is None:
            return
        self.data -= self.grad.data*alpha
    
    def item(self):
        return self.data.item()
    
    def float(self):
        return Tensor(self.data.astype(np.float), autograd=self.autograd)

    def eval(self):
        self.is_eval = True
    
    def train(self):
        self.is_eval = False
    
    def copyfrom(self, other):
        assert self.grad is None, "Can not copy tensor in place during backward"
        assert self.data.shape == other.shape, "The source & target tensor shape should be same"
        assert isinstance(other, np.ndarray), "The type of target should be np.ndarray"

        np.copyto(self.data, other)

    def argmax(self, dim=0):
        return Tensor(self.data.argmax(dim))
    
    def dim(self):
        return self.data.ndim
    
    def size(self, ind):
        assert ind < self.data.ndim, 'index out of data ndim'
        return self.data.shape[ind]
    
    def __str__(self):
        return str(self.data.__str__())

    def __repr__(self):
        return str(self.data.shape)
    
    def __getitem__(self, ind):
        if isinstance(ind, int) or isinstance(ind, slice): # n是索引 or 切片
            return self.data[ind]
        elif isinstance(ind, tuple):
            a_ind, b_ind = ind
            return self.data[a_ind, b_ind]
            