import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torch as t 

train_ds = tv.datasets.MNIST(root='datasets', train=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)  #60000  # bs=32 is good enough

test_ds = tv.datasets.MNIST(root='datasets', train=False, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=True)    #10000

def weight_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()

class MLPNet(nn.Module):
    def __init__(self, layers, num_class):
        super(MLPNet, self).__init__()
        num = len(layers)
        linear_layers = []
        for i in range(num-1):
            linear_layers.append(nn.Linear(layers[i], layers[i+1]))
            weight_init(linear_layers[-1], 0, 0.01)
            linear_layers.append(nn.ReLU())
            
        linear_layers.append(nn.Linear(layers[num-1], num_class))
        self._layers = nn.Sequential(*linear_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output = self._layers(x)
        output = F.log_softmax(output, dim=1)
        return output
        
def calc_precision(gt, pred):
    pred = pred.argmax(dim=1)
    return t.mean((gt == pred).float())

def test(model, criterion, test_dataloader):
    model.eval()
    loss = 0
    precision = 0.0
    num = len(test_dataloader)
    for (x, y) in test_dataloader:
        output = model(x)
        cur_loss = criterion(output, y)
        cur_pre = calc_precision(y, output)

        loss += cur_loss.item()
        precision += cur_pre.item()
    model.train()
    return loss/num, precision/num

def train(epoch, model, criterion, optimizer, train_dataloader):
    for idx in range(epoch):
        loss = 0
        precision = 0.0
        num = len(train_dataloader)
        for (x, y) in train_dataloader:
            
            optimizer.zero_grad()
            output = model(x)
            cur_loss = criterion(output, y)
            cur_loss.backward()
            optimizer.step()

            cur_pre = calc_precision(y, output)

            loss += cur_loss.item()
            precision += cur_pre.item()
    
        test_loss, test_precision = test(mlp_net, criterion, test_dataloader)
        print('epoch %d, in train, loss: %.4f, precision: %.4f, in test, loss: %.4f, precision: %.4f' % (idx, loss/num, precision/num, test_loss, test_precision))


classes_num = 10
hidden_size = 128
h, w = 28, 28
alpha = 0.01

# setup the network
mlp_net = MLPNet(layers=[h*w, hidden_size], num_class=classes_num)
# we only need cross entropy loss, log softmax will be done in mplnet forward function
criterion = nn.NLLLoss()
# no momentum means poor performance(may drop 3 percent), momentum=0.9 is good enough
# the best test precision is 0.9801
sgd = t.optim.SGD(mlp_net.parameters(), lr=alpha, momentum=0.9)
epoch = 15
train(epoch, mlp_net, criterion, sgd, train_dataloader)
        
        
        
        
        
        
        
        