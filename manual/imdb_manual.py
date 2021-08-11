import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

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
        self.vocab = {}
        
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
        
        self.reviews = raw_reviews
        self.labels = raw_labels
    
    def get_vocab_len(self):
        return len(self.vocab)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        r = [self.vocab[w] for w in self.reviews[idx]]
        r = np.array(list(set(r)))  # remove dups
        l = 1 if self.labels[idx]=='positive' else 0
        
        return r, l

batch_size = 1
train_ds = imdb_reviews(reviews_path, labels_path, True)
train_ds.parse_reviews()
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  #20000

test_ds = imdb_reviews(reviews_path, labels_path, False)
test_ds.parse_reviews()
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)    #5000

print('we have %d train samples, and %d test samples' % (len(train_dataloader), len(test_dataloader)))

epochs = 10
alpha = 0.1
vocab_len = train_ds.get_vocab_len()
hidden_size = 128
layer0_w = np.random.normal(0, 0.01, (vocab_len, hidden_size))
layer1_w = np.random.normal(0, 0.01, (hidden_size, 1))
print('length of vocab is', vocab_len)


def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def calc_acc(pred, gt):
    y = (pred>=0.5).astype(np.int)
    return np.mean(y==gt)

def cross_entropy_loss(pred, gt):
    try:
        loss = -gt*np.log(pred)-(1-gt)*np.log(1-pred)
        return np.mean(loss)
    except RuntimeWarning:
        import ipdb
        ipdb.set_trace()
        print('Oh shit! I am in trouble!')

def cross_entropy_loss2(pred, gt):
    loss = 0.0
    for one_pred, one_gt in zip(pred, gt):
        cur_loss = 0
        if one_gt == 1:
            if one_pred == 0:
                one_pred += 1e+7
            cur_loss = -1*np.log(one_pred)
        else:
            if 1-one_pred == 0:
                one_pred -= 1e-7
            cur_loss = -1*np.log(1-one_pred)
        loss += cur_loss

    return np.mean(loss)


def forward(x):
    word_embed = []
    for one in x:
        w = np.mean(layer0_w[one], axis=0)
        word_embed.append(w)
    word_embed = np.stack(word_embed)

    pred = word_embed.dot(layer1_w)
    pred = sigmoid(pred)
    
    return pred, word_embed

def backward(x, word_embed, pred, gt):
    num = pred.shape[0]
    delta = (pred-gt.reshape(num,1))/num
    layer1_delta = word_embed.T.dot(delta)
    
    w_delta = delta.dot(layer1_w.T)
    
    return layer1_delta, w_delta

def step(layer1_delta, w_delta, x):
    global layer0_w, layer1_w
    layer1_w = layer1_w - alpha*layer1_delta
    
    for ind, one in enumerate(x):
        layer0_w[one] = layer0_w[one] - alpha*w_delta[ind]
        
def test():
    loss = 0
    accuracy = 0
    for x,y in test_dataloader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        
        pred, _ = forward(x)
        loss += cross_entropy_loss2(pred, y)
        accuracy += calc_acc(pred, y)
    
    return loss/len(test_dataloader), accuracy/len(test_dataloader)

for epoch in range(epochs):
    loss = 0
    accuracy = 0
    for idx, (x,y) in enumerate(train_dataloader):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        
        pred, word_embed = forward(x)
        loss += cross_entropy_loss2(pred, y)
        accuracy += calc_acc(pred, y)
        
        layer1_delta, w_delta = backward(x, word_embed, pred, y)
        step(layer1_delta, w_delta, x)
    
    # do test
    test_loss, test_acc = test()
    print('in epoch %d, train loss: %.4f, train acc: %.4f, test loss: %.4f, test acc: %.4f' % \
          (epoch, loss/(idx+1), accuracy/(idx+1), test_loss, test_acc))