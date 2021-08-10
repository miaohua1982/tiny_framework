import numpy as np

class QA_Dataset:
    def __init__(self, path):
        self.path = path
        self.start_token = '<START>'
        self.pad_token = '<PAD>'
        self.word2ind = {}
        self.ind2word = {}
        self.corpus = []
    
    def remove_noneed(self, r):
        no_needs = ['0','1','2','3','4','5','6','7','8','9','\n','\t1', '\t2','\t3','\t4','\t5''\t6','\t7','\t8','\t9', '.','?','\t']
        for n in no_needs:
            r = r.replace(n, '')
        return r

    def parse(self):
        f = open(self.path)
        raw = f.readlines()
        f.close()
        self.corpus = [self.remove_noneed(r).split(' ')[1:] for r in raw]
        
        # add pad & start token
        ind = 0
        self.word2ind[self.pad_token] = ind
        self.ind2word[ind] = self.pad_token
        ind = 1
        self.word2ind[self.start_token] = ind
        self.ind2word[ind] = self.start_token
        ind = 2
        
        self.max_len = 0
        for sent in self.corpus:
            if self.max_len < len(sent):
                self.max_len = len(sent)
                
            for w in sent:
                if w not in self.word2ind:
                    self.word2ind[w] = ind
                    self.ind2word[ind] = w
                    ind += 1
        self.max_len += 1  # add start token
        np.random.shuffle(self.corpus)
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sent = self.corpus[idx]
        sent_inds = np.array([self.word2ind[self.start_token]]+[self.word2ind[self.pad_token]]*(self.max_len-1-len(sent))+[self.word2ind[w] for w in sent])
        return sent_inds
    
    def get_sent(self, idx):
        return self.corpus[idx]
    
    def get_vob_len(self):
        return len(self.word2ind)
    
    def get_start_token(self):
        return self.start_token
