import codecs
import copy
import phe
import numpy as np
from framework.tensor import Tensor
from framework.layer import EmbeddingLayer
from framework.optimizer import SGD
from framework.loss import MSELoss

def to_indices(input, word2ind, l=500):
    indices = []
    for one in input:
        if len(one)<l:
            inds = [word2ind[w] for w in one]+[word2ind['<unk>']]*(l-len(one))
            indices.append(inds)
    return indices

def preprocess():
    with codecs.open('datasets/spam.txt',"r",encoding='utf-8',errors='ignore') as f:
        raw = f.readlines()
    vocab, spam, ham = (set(["<unk>"]), list(), list()) 
    for row in raw:
        spam.append(set(row[:-2].split(" ")))  # -2 means remove last 2 words: one space & \n
        for word in spam[-1]:
            vocab.add(word)

    with codecs.open('datasets/ham.txt',"r",encoding='utf-8',errors='ignore') as f:
        raw = f.readlines()
    for row in raw:
        ham.append(set(row[:-2].split(" ")))
        for word in ham[-1]:
            vocab.add(word)

    vocab = list(vocab)
    word2ind = {}
    for ind, word in enumerate(vocab):
        word2ind[word] = ind

    spam_ds = to_indices(spam, word2ind)
    ham_ds = to_indices(ham, word2ind)
    print('spam_ds:', len(spam_ds), 'ham_ds:', len(ham_ds))

    test_ds_len = 1000
    train_spam_ds = spam_ds[:-test_ds_len]
    train_ham_ds = ham_ds[:-test_ds_len]

    test_spam_ds = spam_ds[test_ds_len:]
    test_ham_ds = ham_ds[test_ds_len:]

    # let spam&ham to be same length
    train_ds = []
    for i in range(max(len(train_spam_ds), len(train_ham_ds))):
        train_ds.append(train_spam_ds[i%len(train_spam_ds)]+[1])        
        train_ds.append(train_ham_ds[i%len(train_ham_ds)]+[0])

    np.random.shuffle(train_ds)
    train_target_ds = np.array(train_ds)[:,[-1]]
    train_ds = np.array(train_ds)[:,:-1]
    
    test_ds = []
    test_target_ds = []
    for i in range(test_ds_len):
        test_ds.append(test_spam_ds[i])
        test_target_ds.append([1])
        
        test_ds.append(test_ham_ds[i])
        test_target_ds.append([0])

    return train_ds, train_target_ds, test_ds, test_target_ds, word2ind, vocab

def train(train_ds, train_target_ds, model, batch_size, epochs, word2ind):
    criterion = MSELoss()
    optimizer = SGD(model.get_parameters(), alpha=0.01)
    n_batches = len(train_ds)//batch_size
    counter = 0
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i in range(n_batches):
            x = train_ds[i*batch_size:(i+1)*batch_size]
            y = train_target_ds[i*batch_size:(i+1)*batch_size]
            input = Tensor(x, autograd=True)
            target = Tensor(y, autograd=True)
            
            pred = model.forward(input).sum(1).sigmoid()
            loss = criterion(pred, target)
            acc = ((pred.data>=0.5) == target.data).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.embedding_weights.data[word2ind['<unk>']] *= 0  # unknown index must be stay 0, it is a trick, just have a thought why we have to set unk's weight to be zero

            total_loss += loss[0]
            total_acc += acc
        
        print('In %d epoch, loss %.4f, acc %.4f' %(epoch, total_loss/n_batches, total_acc/n_batches))

        return model
        
def test(test_ds, test_target_ds, model):
    criterion = MSELoss()
    input = Tensor(test_ds)
    target = Tensor(test_target_ds)

    pred = model.forward(input).sum(1).sigmoid()
    loss = criterion(pred, target)
    acc = ((pred.data>=0.5) == target.data).mean()
    
    return loss[0], acc

def train_and_encrypt(input, target, model, batch_size, word2ind, pubkey):
    new_model = train(input, target, model, batch_size, 1, word2ind)

    encrypted_weights = list()
    for val in new_model.embedding_weights.data[:,0]:
        encrypted_weights.append(pubkey.encrypt(val))
    ew = np.array(encrypted_weights).reshape(new_model.embedding_weights.data.shape)
    
    return ew

def vallina_main():
    train_ds, train_target_ds, test_ds, test_target_ds, word2ind, vocab = preprocess()
    model = EmbeddingLayer(len(vocab), 1)
    model.embedding_weights.data *= 0
    epochs = 3
    for i in range(epochs):
        model = train(train_ds, train_target_ds, model, 500, 1, word2ind)
        model.embedding_weights.data[word2ind['<unk>']] *= 0

        test_loss, test_acc = test(test_ds, test_target_ds, model)
        print('In %d epoch, test loss %.4f, acc %.4f' %(i, test_loss, test_acc))

def federate_main():
    train_ds, train_target_ds, test_ds, test_target_ds, word2ind, vocab = preprocess()
    model = EmbeddingLayer(len(vocab), 1)
    model.embedding_weights.data *= 0    
    epochs = 3
    for i in range(epochs):
        bob = (train_ds[0:1000], train_target_ds[0:1000])
        alice = (train_ds[1000:2000], train_target_ds[1000:2000]) 
        sue = (train_ds[2000:], train_target_ds[2000:])
        
        bob_model = train(bob[0], bob[1], copy.deepcopy(model), 500, 1, word2ind)
        alice_model = train(alice[0], alice[1], copy.deepcopy(model), 500, 1, word2ind)
        sue_model = train(sue[0], sue[1], copy.deepcopy(model), 500, 1, word2ind)
        
        model.embedding_weights.data = (bob_model.embedding_weights.data+alice_model.embedding_weights.data+sue_model.embedding_weights.data)/3
        print(model.embedding_weights.data[word2ind['<unk>']])
        model.embedding_weights.data[word2ind['<unk>']] *= 0
        test_loss, test_acc = test(test_ds, test_target_ds, model)
        print('In %d round federated learning, test loss %.4f, acc %.4f' %(i, test_loss, test_acc))

def enc_federate_main():
    train_ds, train_target_ds, test_ds, test_target_ds, word2ind, vocab = preprocess()
    public_key, private_key = phe.generate_paillier_keypair(n_length=128)
    model = EmbeddingLayer(len(vocab), 1)
    model.embedding_weights.data *= 0    
    epochs = 3
    for i in range(epochs):
        bob = (train_ds[0:1000], train_target_ds[0:1000])
        alice = (train_ds[1000:2000], train_target_ds[1000:2000]) 
        sue = (train_ds[2000:], train_target_ds[2000:])
        
        bob_weights = train_and_encrypt(bob[0], bob[1], copy.deepcopy(model), 500, word2ind, public_key)
        alice_weights = train_and_encrypt(alice[0], alice[1], copy.deepcopy(model), 500, word2ind, public_key)
        sue_weights = train_and_encrypt(sue[0], sue[1], copy.deepcopy(model), 500, word2ind, public_key)
        
        total_weights = bob_weights+alice_weights+sue_weights
        raw_values = list()
        for val in total_weights.flatten():
            raw_values.append(private_key.decrypt(val))
        model.embedding_weights.data = np.array(raw_values).reshape(model.embedding_weights.data.shape)/3

        test_loss, test_acc = test(test_ds, test_target_ds, model)
        print('In %d round federated learning, test loss %.4f, acc %.4f' %(i, test_loss, test_acc))


if __name__ == '__main__':
    #vallina_main()
    enc_federate_main()