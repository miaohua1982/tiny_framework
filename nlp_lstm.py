from os import sep
import string
import numpy as np
from torch.utils.data import dataset
from framework.tensor import Tensor
from framework.rnn import RNN_Model
from framework.loss import CrossEntropyLoss
from framework.optimizer import SGD

def train(epoches, datasets, model, criterion, optim, hidden_size):
    batch_size = 16
    bptt = 25

    n_batches = len(datasets)//batch_size
    datasets = datasets[:batch_size*n_batches]
    datasets = datasets.reshape(batch_size, n_batches).transpose()
    input_batched_indices = datasets[0:-1]
    target_batched_indices = datasets[1:]
    n_bptt = int(((n_batches-1) / bptt))
    input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
    target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
    

    for epoch in range(epoches):
        total_acc = 0
        total_loss = 0
        counter = 0
                    
        cur_loss_arr = []
        for i in range(0, n_bptt):
            optim.zero_grad()
            hidden = Tensor(np.zeros((batch_size, hidden_size)), autograd=True)
            for j in range(bptt):
                x = input_batches[i, j]
                y = target_batches[i, j]
            
                input = Tensor(x, autograd=True)
                target = Tensor(y, autograd=True)

                pred, hidden = model.forward(input, hidden)
                loss = criterion(pred, target)

                if j == 0:
                    cur_loss_arr.append(loss)
                else:
                    cur_loss_arr.append(loss+cur_loss_arr[-1])
            
            acc = (pred.data.argmax(axis=1) == target.data).mean()
            total_acc += acc
            cur_loss = cur_loss_arr[-1]
            total_loss += cur_loss[0]/bptt
            counter += 1
            
            cur_loss.backward()
            optim.step()

        optim.decay_lr()
            
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
    epoches = 10
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
    optim = SGD(parameters=model.get_parameters(), alpha=0.005, decay=0.99)

    train(epoches, inputs, model, criterion, optim, hidden_size)
    predict_byinit(word2ind, ind2word, 'T', hidden_size, model)

if __name__ == '__main__':
    # it is a char2char model, note ** char generates char **, not word by word
    main()


