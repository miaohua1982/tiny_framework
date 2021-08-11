from torch.utils.data import DataLoader
import numpy as np
import random
import sys
import time

reviews_path = 'datasets/IMDB/reviews.txt'
labels_path = 'datasets/IMDB/labels.txt'

class imdb_reviews:
    def __init__(self, reviews_path, labels_path, train):
        self.reviews_path = reviews_path
        self.labels_path = labels_path
        self.is_train = train
        self.test_num = 1000
        
        self.reviews = []
        self.labels = []
        self.word2ind = []
        self.vocab = {}
        self.vocab_reverse = {}
        
    def parse_reviews(self):
        f = open(self.reviews_path)
        raw_reviews = f.readlines()
        f.close()
        
        f = open(self.labels_path)
        raw_labels = f.readlines()
        f.close()
        
        raw_reviews = [r.split() for r in raw_reviews]
        ind = 0
        for r in raw_reviews:
            for w in r:
                if w not in self.vocab:
                    self.vocab[w] = ind
                    self.vocab_reverse[ind] = w
                    ind += 1
                    
        if self.is_train:
            raw_reviews = raw_reviews[:-1*self.test_num]
        else:
            raw_reviews = raw_reviews[-1*self.test_num:]
        
        if self.is_train:
            raw_labels = raw_labels[:-1*self.test_num]
        else:
            raw_labels = raw_labels[-1*self.test_num:]
        raw_labels = [r[:-1] for r in raw_labels]
        
        self.word2ind = list(self.vocab_reverse.keys())
        self.reviews = raw_reviews
        self.labels = raw_labels
    
    def get_vocab_len(self):
        return len(self.vocab)
    
    def get_word_ind(self, word):
        return self.vocab[word]
    
    def get_ind_word(self, ind):
        return self.vocab_reverse[ind]
    
    def gen_random_word(self, win=2):
        targets = random.choices(self.word2ind, k=win)
        return targets
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        r = np.array([self.vocab[w] for w in self.reviews[idx]])
        l = 1 if self.labels[idx]=='positive' else 0
        
        return r, l


batch_size = 1
train_ds = imdb_reviews(reviews_path, labels_path, True)
train_ds.parse_reviews()
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  #24000

test_ds = imdb_reviews(reviews_path, labels_path, False)
test_ds.parse_reviews()
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)    #1000

print('we have %d train samples, and %d test samples' % (len(train_dataloader), len(test_dataloader)))

alpha = 0.05
win = 2
negative = 5
target_y = np.zeros((1,negative+1))
target_y[0,0] = 1

vocab_len = train_ds.get_vocab_len()
hidden_size = 128

layer0_w = np.random.normal(0, 0.02, (vocab_len, hidden_size))-0.01
layer1_w = np.random.normal(0, 0.02, (vocab_len, hidden_size))-0.01
print('length of vocab is', vocab_len)

def calc_similar(w_vec1, words_embed, topk=10):
    similar_ratio = {}
    for ind, one_vec in enumerate(words_embed):
        s = np.sum((w_vec1-one_vec)**2)**0.5
        similar_ratio[ind] = s
    similar = sorted(similar_ratio.items(), key=lambda x:x[1])    
    return similar[:topk]

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def forward(left, right, target_sample):
    word_embed = np.mean(layer0_w[left+right], axis=0, keepdims=True)

    pred = word_embed.dot(layer1_w[target_sample].T)
    pred = sigmoid(pred)
    
    return pred, word_embed

def backward(x, word_embed, pred, gt):
    num = pred.shape[0]
    delta = (pred-gt)/num
    layer1_delta = word_embed.T.dot(delta)  # hidden_size*(w+1)
    
    w_delta = delta.dot(layer1_w[x])        # 1*hidden_size
    
    return layer1_delta.T, w_delta

def step(layer1_delta, w_delta, left, right, target_sample):
    global layer0_w, layer1_w
    
    layer1_w[target_sample] -= alpha*layer1_delta
    layer0_w[left+right] -= alpha*w_delta

epochs = 2
train_ds_num = len(train_dataloader)
for epoch in range(epochs):
    total_loss = 0
    total_accuracy = 0
    total_counter = 0
    for idx, (x,y) in enumerate(train_dataloader):
        x = x.cpu().numpy().tolist()[0]
        y = y.cpu().numpy()
        sentence_len = len(x)
        
        loss = 0
        accuracy = 0
        counter = 0
        for i, target_word in enumerate(x):
            left = x[max(0, i-win):i]
            right = x[i+1:min(sentence_len, i+1+win)]
            
            target_sample = [target_word]+train_ds.gen_random_word(negative)
            #counter += 1
            pred, word_embed = forward(left, right, target_sample)
            #loss += cross_entropy_loss2(pred, target_y)
            #accuracy += calc_acc(pred, target_y)
        
            layer1_delta, w_delta = backward(target_sample, word_embed, pred, target_y)
            step(layer1_delta, w_delta, left, right, target_sample)

        #print('in cur sentence, train loss: %.4f, train acc: %.4f' % (loss/counter, accuracy/counter))
        
        #total_counter += counter
        #total_loss += loss
        #total_accuracy += accuracy
        
        if (idx+1) % 200 == 0:
            sys.stdout.write('\rCurrent Progress:'+str((idx+epoch*train_ds_num)/(epochs*train_ds_num)))
print('')
    #print('in epoch %d, train loss: %.4f, train acc: %.4f' % (epoch, total_loss/total_counter, total_accuracy/total_counter))

# check for similar word
#---------------------------------------------------------------------------------------
def calc_similar(w_vec1, words_embed, topk=10):
    similar_ratio = {}
    for ind, one_vec in enumerate(words_embed):
        s = np.sum((w_vec1-one_vec)**2)**0.5
        similar_ratio[ind] = s
    similar = sorted(similar_ratio.items(), key=lambda x:x[1])    
    return similar[:topk]

word = 'beautiful'
word_ind = train_ds.get_word_ind(word)
word_vec = layer0_w[word_ind,:]
similar_topk = calc_similar(word_vec, layer0_w)
print(word,'\' similar top 10 is:')
for one_similar in similar_topk:
    print(train_ds.get_ind_word(one_similar[0]),':',one_similar[1])
print('-'*64)
#--------------------------------------------------------------------------------------

#What is King-Man+Woman ?
def analogy(words_embed, positive=['terrible','good'], negative=['bad'], topk=10):
    query_word_embed = np.zeros((1, hidden_size))
    for one_word in positive:
        word_ind = train_ds.get_word_ind(one_word)
        query_word_embed += words_embed[word_ind]
    
    for one_word in negative:
        word_ind = train_ds.get_word_ind(one_word)
        query_word_embed -= words_embed[word_ind]
    
    similar_ratio = {}
    for ind, one_vec in enumerate(words_embed):
        s = np.sum((query_word_embed-one_vec)**2)**0.5
        similar_ratio[ind] = s
    similar = sorted(similar_ratio.items(), key=lambda x:x[1])    
    return similar[:topk]

#terrible+good-bad = ?
similar = analogy(layer0_w, ['terrible','good'], ['bad'])
for one in similar:
    word = train_ds.get_ind_word(one[0]) 
    print(word,' '*(15-len(word)),':  ', one[1])
print('-'*64)
#伊丽莎白＋他－她＝？
similar = analogy(layer0_w, ['elizabeth','he'], ['she'])
for one in similar:
    word = train_ds.get_ind_word(one[0]) 
    print(word,' '*(15-len(word)),':  ', one[1])