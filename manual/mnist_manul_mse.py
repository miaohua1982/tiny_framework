import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

train_ds = tv.datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)  #60000

test_ds = tv.datasets.MNIST(root='datasets', train=False, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=True)    #10000

classes_num = 10
hidden_size = 128
h, w = 28, 28

layer1_w = np.random.normal(0, 0.01, (h*w,hidden_size))
layer1_b = np.zeros(hidden_size)

layer2_w = np.random.normal(0, 0.01, (hidden_size,classes_num))
layer2_b = np.zeros(classes_num)


def calc_precision(gt, pred):
    pred = pred.argmax(axis=1)
    return np.mean(gt == pred)

def mse(gt, pred):
    y = np.zeros((gt.shape[0], 10))
    y[np.arange(gt.shape[0]), gt] = 1
    
    return np.mean((y-pred)**2)

def relu(x):
    return np.where(x>=0, x, 0)
    
def forward(x):
    x.resize(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    # layer1
    layer1_output = relu(np.matmul(x, layer1_w)+layer1_b)
    # layer2
    layer2_output = np.matmul(layer1_output, layer2_w)+layer2_b
    
    return layer2_output, layer1_output
    
def backward(gt, pred, layer1_output, x):
    num = gt.shape[0]
    y = np.zeros((num, 10))
    y[np.arange(num), gt] = 1
    diff_loss = (pred-y)*2/num                                      # 16*10
    
    diff_w2 = np.matmul(layer1_output.T, diff_loss)                 # 16*20, 16*10 = 20*10
    diff_b2 = np.matmul(np.ones(num), diff_loss)                    #  16, 16*10   = 10
    
    diff_loss = np.matmul(diff_loss, layer2_w.T)           # 16*10, 20*10 = 16*20
    diff_loss = np.where(layer1_output>0, 1, 0)*diff_loss # relu difference
    diff_w1 = np.matmul(x.T, diff_loss)                    # 
    diff_b1 = np.matmul(np.ones(num), diff_loss)           # 16, 16*20 = 20
    
    return diff_w1, diff_b1, diff_w2, diff_b2
    
def step_sgd(lr, diff_w1, diff_b1, diff_w2, diff_b2):
    global layer1_w, layer1_b, layer2_w, layer2_b
    layer1_w = layer1_w - lr*diff_w1
    layer1_b = layer1_b - lr*diff_b1

    layer2_w = layer2_w - lr*diff_w2
    layer2_b = layer2_b - lr*diff_b2

def test(test_dataloader):
    loss = 0
    precision = 0.0
    num = len(test_dataloader)
    for (x, y) in test_dataloader:
        x = x.detach().cpu().numpy().copy()
        y = y.detach().cpu().numpy().copy()
        
        output, _ = forward(x)
        
        cur_loss = mse(y, output)
        cur_pre = calc_precision(y, output)
        loss += cur_loss
        precision += cur_pre

    return loss/num, precision/num

def train(epochs, lr, train_dataloader):
    for epoch in range(epochs):
        loss = 0
        precision = 0.0
        num = len(train_dataloader)
        for idx, (x, y) in enumerate(train_dataloader):
            #print('in epoch %d, batch %d' % (epoch, idx))
            x = x.detach().cpu().numpy().copy()
            y = y.detach().cpu().numpy().copy()
            
            output, layer1_output = forward(x)

            cur_loss = mse(y, output)
            cur_pre = calc_precision(y, output)
            loss += cur_loss
            precision += cur_pre

            diff_w1, diff_b1, diff_w2, diff_b2 = backward(y, output, layer1_output, x)
            step_sgd(lr, diff_w1, diff_b1, diff_w2, diff_b2)
            
        test_loss, test_precision = test(test_dataloader)

        print('epoch %d, in train, loss: %.4f, precision: %.4f, in test, loss: %.4f, precision: %.4f' % (epoch, loss/num, precision/num, test_loss, test_precision))

epoch = 20
lr = 0.01
train(epoch, lr, train_dataloader) # we get best precision 0.9688
