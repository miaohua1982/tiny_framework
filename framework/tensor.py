import numpy as np

class Tensor(object):
    def __init__(self, data, autograd=False, creator=None, create_op=None, id=None):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.creator = creator
        self.create_op = create_op
        self.autograd = autograd
        self.grad = None
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
        
    def sum(self, dim):
        assert self.data.ndim>dim, 'axis %d is out of bounds for array of dimension %d' % (dim, self.data.ndim)
        if self.autograd:
            return Tensor(self.data.sum(dim), autograd=True, creator=(self,), create_op='sum_'+str(dim))
        return Tensor(self.data.sum(dim))
    
    def mean(self, dim):
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
    
    def relu(self):
        if self.autograd:
            return Tensor(np.where(self.data>0, self.data, 0), autograd=True, creator=(self,), create_op='relu')
        return Tensor(np.where(self.data>0, self.data, 0))
    
    def sigmoid(self):
        new = 1/(1+np.exp(-self.data))
        if self.autograd:
            return Tensor(new, autograd=True, creator=(self,), create_op='sigmoid')
        return Tensor(new)
    
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
                self.creator[0].backward(self.grad*factor, self)
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

    def zero_grad(self):
        self.grad = None
       
    def step(self, alpha):
        if self.grad is None:
            return
        self.data -= self.grad.data*alpha
    
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
            