import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

classes_num = 10
hidden_size = 128
h, w = 28, 28
batch_size = 32
iteration_num = 10

train_ds = tv.datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  #60000

test_ds = tv.datasets.MNIST(root='datasets', train=False, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)    #10000


layer1_w = np.random.normal(0, 0.01, (h*w,hidden_size))
layer1_wv = np.zeros((h*w,hidden_size))
layer1_b = np.zeros(hidden_size)
layer1_bv = np.zeros(hidden_size)

layer2_w = np.random.normal(0, 0.01, (hidden_size,classes_num))
layer2_wv = np.zeros((hidden_size,classes_num))
layer2_b = np.zeros(classes_num)
layer2_bv = np.zeros(classes_num)

def calc_precision(gt, pred):
    pred = pred.argmax(axis=1)
    return np.mean(gt == pred)

def cross_entropy(gt, pred):
    num = pred.shape[0]
    return np.mean(-1*np.log(pred[np.arange(num), gt]))
    
def null_loss(gt, pred):
    num = pred.shape[0]
    return np.mean(-1*pred[np.arange(num), gt])

def relu(x):
    return np.where(x>=0, x, 0)
    
def softmax(x):
    s = np.exp(x-x.max(axis=1, keepdims=True))
    return s/s.sum(axis=1, keepdims=True)
    
def log_softmax(x):
    s = np.exp(x-x.max(axis=1, keepdims=True))
    return x-x.max(axis=1, keepdims=True)-np.log(s.sum(axis=1, keepdims=True))
    

def forward(x):
    x.resize(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    # layer1
    layer1_output = relu(np.matmul(x, layer1_w)+layer1_b)
    # layer2
    layer2_output = np.matmul(layer1_output, layer2_w)+layer2_b
    final_output = softmax(layer2_output)
    
    return final_output, layer1_output
    
def backward(gt, pred, layer1_output, x):
    # cross_entropy difference  -1/pred
    # softmax difference pred(1-pred)
    # layer2 w difference x, b difference 1
    num = gt.shape[0]
    #diff_loss = np.zeros((num, classes_num))
    diff_loss = pred.copy()
    #diff_loss -= 1
    diff_loss[np.arange(num), gt] = pred[np.arange(num), gt]-1           # 16*10
    #diff_loss /= num

    diff_w2 = np.matmul(layer1_output.T, diff_loss)                      # 16*20, 16*10 = 20*10
    diff_b2 = np.matmul(np.ones(num), diff_loss)                         # 16, 16*10   = 10
    
    diff_loss = np.matmul(diff_loss, layer2_w.T)             # 16*10, 20*10 = 16*20
    diff_loss = np.where(layer1_output>0, 1, 0)*diff_loss    # relu difference
    diff_w1 = np.matmul(x.T, diff_loss)                      # 
    diff_b1 = np.matmul(np.ones(num), diff_loss)             # 16, 16*20 = 20
    
    return diff_w1, diff_b1, diff_w2, diff_b2
    
def step_sgd(lr, diff_w1, diff_b1, diff_w2, diff_b2):
    global layer1_w, layer1_b, layer2_w, layer2_b
    layer1_w = layer1_w - lr*diff_w1
    layer1_b = layer1_b - lr*diff_b1

    layer2_w = layer2_w - lr*diff_w2
    layer2_b = layer2_b - lr*diff_b2

# momentum sgd
def step_momentum(lr, diff_w1, diff_b1, diff_w2, diff_b2, momentum=0.9):
    global layer1_w, layer1_b, layer2_w, layer2_b, layer1_wv, layer1_bv, layer2_wv, layer2_bv
    layer1_wv = momentum*layer1_wv+lr*diff_w1
    layer1_bv = momentum*layer1_bv+lr*diff_b1
    layer2_wv = momentum*layer2_wv+lr*diff_w2
    layer2_bv = momentum*layer2_bv+lr*diff_b2

    layer1_w = layer1_w - layer1_wv
    layer1_b = layer1_b - layer1_bv

    layer2_w = layer2_w - layer2_wv
    layer2_b = layer2_b - layer2_bv

def test(test_dataloader):
    loss = 0
    precision = 0.0
    num = len(test_dataloader)
    for (x, y) in test_dataloader:
        x = x.detach().cpu().numpy().copy()
        y = y.detach().cpu().numpy().copy()
        
        output, _ = forward(x)
        
        cur_loss = cross_entropy(y, output)
        cur_pre = calc_precision(y, output)
        loss += cur_loss
        precision += cur_pre

    return loss/num, precision/num

def train(epochs, lr, train_dataloader):
    for epoch in range(epochs):
        loss = 0
        precision = 0.0
        num = 0
        for idx, (x, y) in enumerate(train_dataloader):
            #print('in epoch %d, batch %d' % (epoch, idx))
            x = x.detach().cpu().numpy().copy()
            y = y.detach().cpu().numpy().copy()
            num += 1
            
            output, layer1_output = forward(x)

            cur_loss = cross_entropy(y, output)
            cur_pre = calc_precision(y, output)
            loss += cur_loss
            precision += cur_pre

            diff_w1, diff_b1, diff_w2, diff_b2 = backward(y, output, layer1_output, x)
            step_sgd(lr, diff_w1, diff_b1, diff_w2, diff_b2)
        
        test_loss, test_precision = test(test_dataloader)

        print('epoch %d, in train, loss: %.4f, precision: %.4f, in test, loss: %.4f, precision: %.4f' % (epoch, loss/num, precision/num, test_loss, test_precision))

# the best test precision is 0.9814 use common sgd
# the best test precision is 0.9776 use momentum

epoch = 20
lr = 0.01
train(epoch, lr, train_dataloader)