from torch.utils.data import DataLoader
import numpy as np

from utils.QA_Dataset import QA_Dataset
from framework.tensor import Tensor
from framework.rnn import RNN_Model
from framework.loss import CrossEntropyLoss
from framework.optimizer import SGD

def dataset_show(qa_ds):
    batch_size = 16
    train_dataloader = DataLoader(qa_ds, batch_size=batch_size, shuffle=True)
    batch_x = iter(train_dataloader).next()
    print(batch_x)
    print('-'*64)
    for idx, one_sent in enumerate(batch_x):
        print(idx, sep ='.', end=' ')
        for one_word in one_sent:
            print(qa_ds.ind2word[one_word.item()],sep=' ', end=' ')
        print('')
    print('-'*64)

def train(epoches, qa_ds, train_dataloader, model, criterion, optim, batch_size, hidden_size):
    for epoch in range(epoches):
        total_acc = 0
        total_loss = 0
        counter = 0
        for one_bs in train_dataloader:
            hidden = Tensor(np.zeros((batch_size, hidden_size)), autograd=True)
            
            for i in range(qa_ds.max_len-1):
                x = one_bs[:,i]
                y = one_bs[:,i+1]

                input = Tensor(x.detach().cpu().numpy(), autograd=True)
                target = Tensor(y.detach().cpu().numpy(), autograd=True)

                pred, hidden = model.forward(input, hidden)

                optim.zero_grad()    
                
                loss = criterion(pred, target)
                acc = (pred.data.argmax(axis=1) == target.data).mean()
                total_loss += loss[0]
                total_acc += acc
                counter += 1
            
                loss.backward()
                optim.step()
                
                
        print('In epoch %d, nn gets loss %.4f, acc %.4f' % (epoch, total_loss/counter, total_acc/counter))

def test_predict_onebyone(qa_ds, hidden_size, model):
    print('-'*64)
    test_num = 5
    for i in range(test_num):
        sent = qa_ds[i]
        print('Gt sentence:', ' '.join(qa_ds.get_sent(i)))
        pred_sent = ""
        hidden = Tensor(np.zeros((1, hidden_size)), autograd=True)
        for j in range(len(sent)-1):
            x = sent[j]
            gt = sent[j+1]
            
            input = Tensor([x], autograd=True)
            pred, hidden = model.forward(input, hidden)
            pred_y = pred.data.argmax(axis=1)[0]
            print('Get', qa_ds.ind2word[x],'Pred:', qa_ds.ind2word[pred_y], 'Gt:', qa_ds.ind2word[gt])
            pred_sent += qa_ds.ind2word[pred_y] +" "
        print('Pred sentence:', pred_sent)
        print('-'*64)

def test_predict_onebycontext(qa_ds, hidden_size, model):
    print('-'*64)
    test_num = 5
    for i in range(test_num):
        sent = qa_ds[i]
        print('Gt sentence:', ' '.join(qa_ds.get_sent(i)))
        context = ""
        hidden = Tensor(np.zeros((1, hidden_size)), autograd=True)
        for j in range(len(sent)-1):
            x = sent[j]
            gt = sent[j+1]
            context += qa_ds.ind2word[x]+" "
            
            input = Tensor([x], autograd=True)
            pred, hidden = model.forward(input, hidden)
            
        pred_y = pred.data.argmax(axis=1)[0]
        print('Context:', context,'Pred:', qa_ds.ind2word[pred_y], '-- Gt:', qa_ds.ind2word[gt])
        print('-'*64)

def main():
    epoches = 15
    batch_size = 100
    path = 'datasets/qa1_single-supporting-fact_train.txt'
    qa_ds = QA_Dataset(path)
    qa_ds.parse()
    train_dataloader = DataLoader(qa_ds, batch_size=batch_size, shuffle=True)
    dataset_show(qa_ds)

    word_embedding_size = 64
    hidden_size = 64
    vocab_size = qa_ds.get_vob_len()
    model = RNN_Model(word_embedding_size, hidden_size, vocab_size)
    criterion = CrossEntropyLoss()
    optim = SGD(parameters=model.get_parameters(), alpha=0.001)

    train(epoches, qa_ds, train_dataloader, model, criterion, optim, batch_size, hidden_size)
    test_predict_onebyone(qa_ds, hidden_size, model)
    test_predict_onebycontext(qa_ds, hidden_size, model)


if __name__ == '__main__':
    main()