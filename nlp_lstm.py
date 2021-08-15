from os import sep
import string
import numpy as np
from torch.utils.data import dataset
from framework.tensor import Tensor
from framework.rnn import RNN_Model
from framework.loss import CrossEntropyLoss
from framework.optimizer import SGD

def makeup_str(r):
    no_needs = ['0','1','2','3','4','5','6','7','8','9','\n','\t']
    for n in no_needs:
        r = r.replace(n, ' ')
    for s in string.punctuation:
        r = r.replace(s, ' '+s+' ')
    return r

def train(epoches, datasets, word2ind, model, criterion, optim, hidden_size):
    batch_size = 16
    bppt = 25

    n_batches = len(datasets)//batch_size
    datasets = datasets[:batch_size*n_batches]
    datasets = datasets.reshape(n_batches, batch_size)
    n_bppt_bs = n_batches//bppt
    datasets = datasets[:n_bppt_bs*bppt]
    datasets = datasets.reshape(n_bppt_bs, bppt, batch_size)

    for epoch in range(epoches):
        total_acc = 0
        total_loss = 0
        counter = 0
        
        for i in range(0, n_bppt_bs):
            optim.zero_grad()
            hidden = Tensor(np.zeros((bppt, hidden_size)), autograd=True)
            for j in range(batch_size-1):
                x = datasets[i, :, j]
                y = datasets[i, :, j+1]
            
                input = Tensor(x, autograd=True)
                target = Tensor(y, autograd=True)

                pred, hidden = model.forward(input, hidden)

                loss = criterion(pred, target)
                acc = (pred.data.argmax(axis=1) == target.data).mean()
                total_loss += loss[0]
                total_acc += acc
                counter += 1
            
                loss.backward()
            optim.step()
            
        print('In epoch %d, nn gets loss %.4f, acc %.4f' % (epoch, total_loss/counter, total_acc/counter))

def predict_byinit(word2ind, ind2word, init_char, hidden_size, model, max_len=100):
    print('-'*64)
    print(init_char, sep='', end='')
    input = Tensor([word2ind[init_char]], autograd=True)
    hidden = Tensor(np.zeros((1, hidden_size)), autograd=True)
    for i in range(max_len):
        pred, hidden = model.forward(input, hidden)
        pred_y = pred.data.argmax(axis=1)[0]
        print(ind2word[pred_y], sep='', end='')
        input = Tensor([pred_y], autograd=True)

    print('')
    print('-'*64)

def main():
    epoches = 100
    path = 'datasets/shakespear.txt'
    f = open(path, 'r')
    raw = f.read()
    f.close()

    vocab = list(set(raw))
    word2ind = {}
    ind2word = {}

    ind = 0
    for one_word in vocab:
        word2ind[one_word] = ind
        ind2word[ind] = one_word
        ind += 1
    inputs = np.array(list(map(lambda x: word2ind[x], raw)))

    word_embedding_size = 512
    hidden_size = 512
    vocab_size = len(vocab)
    model = RNN_Model(word_embedding_size, hidden_size, vocab_size)
    criterion = CrossEntropyLoss()
    optim = SGD(parameters=model.get_parameters(), alpha=0.005)

    train(epoches, inputs, word2ind, model, criterion, optim, hidden_size)
    predict_byinit(word2ind, ind2word, 'T', hidden_size, model)

if __name__ == '__main__':
    # it is a char2char model, note ** char generates char **, not word by word
    main()


