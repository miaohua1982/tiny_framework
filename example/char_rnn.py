import numpy as np
np.random.seed(0)

path = 'datasets/shakespear.txt'
f = open(path, 'r')
raw = f.read()
f.close()

vocab = sorted(list(set(raw)))
word2ind = {}
ind2word = {}

ind = 0
for one_word in vocab:
    word2ind[one_word] = ind
    ind2word[ind] = one_word
    ind += 1
inputs = np.array(list(map(lambda x: word2ind[x], raw)))
print(word2ind)

batch_size = 32
bptt = 16

datasets = inputs
n_batches = len(datasets)//batch_size
datasets = datasets[:batch_size*n_batches]
datasets = datasets.reshape(batch_size, n_batches).transpose()
input_batched_indices = datasets[0:-1]
target_batched_indices = datasets[1:]
n_bptt = int(((n_batches-1) / bptt))
input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
print('input batches', input_batches.shape, 'target batches', target_batches.shape)

def softmax(pred):
    y = np.exp(pred).sum(axis=1, keepdims=True)
    return np.exp(pred)/y

def cross_entropy_loss(pred, gt):
    num = gt.shape[0]
    loss = -np.log(pred[np.arange(num), gt])
    return np.sum(loss)

# previous one may have problem, when pred = 0 or 1
def cross_entropy_loss2(pred, gt):
    loss = 0.0
    num = gt.shape[0]
    for one_pred, one_gt in zip(pred, gt):
        cur_loss = 0
        p = pred[one_gt]
        if p == 0:
            p += 1e+7
            cur_loss = -1*np.log(p)
        loss += cur_loss

    return np.mean(loss)

def cross_entropy_loss_bs(pred, gt):
    L = 0
    for p, g in zip(pred, gt):
        cur_loss = cross_entropy_loss(p, g)
        L += cur_loss
    
    return L

def calc_acc(pred, gt):
    y = pred.argmax(axis=1)
    return np.mean(y == gt)

def calc_acc_bs(pred, gt):
    acc = 0
    for p, g in zip(pred, gt):
        cur_acc = calc_acc(p, g)
        acc += cur_acc
    return acc

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forward(x, h=None):
    preds = []
    hidden_state = []
    
    if h is None:   # h0
        h = np.zeros((batch_size,hidden_size))
    hidden_state.append(h)
    
    for word in x:
        word_vec = word_embed[word].dot(input_layer)
        h = sigmoid(h.dot(transition_layer)+word_vec)
        pred = softmax(h.dot(output_layer))
        
        preds.append(pred)
        hidden_state.append(h)
        
    return np.stack(preds), np.stack(hidden_state)

def backward(preds, hidden_state, inputs, target):
    num = preds.shape[0]
    bs = preds.shape[1]
    vocab_len = preds.shape[2]
    hidden_size = hidden_state.shape[2]
    
    output_layer_delta = np.zeros((hidden_size, vocab_len))
    input_layer_delta = np.zeros((hidden_size, hidden_size))
    transition_layer_delta = np.zeros((hidden_size,hidden_size))
    word_embed_delta = np.zeros((vocab_len, hidden_size))
    h0_delta = np.zeros((bs, hidden_size))
    
    for ind in reversed(range(num)):
        input = inputs[ind]
        pred = preds[ind]
        words = target[ind]
        
        y = np.zeros((bs,vocab_len))
        prev_h = hidden_state[ind]
        cur_h = hidden_state[ind+1]
        y[np.arange(bs), words] = 1
        
        delta = pred - y
        output_layer_delta += cur_h.T.dot(delta)   # shape = hidden_size * vocab
        next_grad = delta.dot(output_layer.T)      # shape = batch_size * hidden_size
        next_grad = next_grad*(cur_h*(1-cur_h))    # shape = batch_size * hidden_size, sigmoid's differ
        transition_layer_delta += prev_h.T.dot(next_grad)  # shape = hidden_size*hidden_size
        word_embed_delta[input] += next_grad.dot(input_layer.T)     # shape = batch_size*hidden_size, input_layer = hidden_size*hidden_size
        input_layer_delta += word_embed[input].T.dot(next_grad)
        
        if ind>0:
            # for previous layers' gradient
            prev_gradient = delta.dot(output_layer.T)   # shape=batch_size*hidden_size
            prev_gradient = prev_gradient*(cur_h*(1-cur_h))
            for cur_ind in range(ind,0,-1):
                input = inputs[cur_ind-1]
                cur_h = hidden_state[cur_ind]
                prev_h = hidden_state[cur_ind-1]
                cur_gradient = prev_gradient.dot(transition_layer.T)
                cur_gradient = cur_gradient*(cur_h*(1-cur_h))
                transition_layer_delta += prev_h.T.dot(cur_gradient)           # shape = hidden_size*hidden_size
                word_embed_delta[input] += cur_gradient.dot(input_layer.T)     # shape = batch_size*hidden_size
                input_layer_delta += word_embed[input].T.dot(cur_gradient)
                prev_gradient = cur_gradient
        
        h0_delta += prev_gradient.dot(transition_layer.T)
        
    return input_layer_delta, output_layer_delta, transition_layer_delta, word_embed_delta, h0_delta

def step(input_layer_delta, output_layer_delta, transition_layer_delta, word_embed_delta, h0_delta):
    global input_layer, output_layer, transition_layer, word_embed, h0
    num = x.shape[0]
    input_layer -= alpha*input_layer_delta/num
    output_layer -= alpha*output_layer_delta/num
    transition_layer -= alpha*transition_layer_delta/num
    word_embed -= alpha*word_embed_delta/num
    h0 -= h0_delta/num

def predict_byinit_random(word2ind, ind2word, init_char, hidden_size, max_len=100):
    s = ""
    
    input = np.array([word2ind[init_char]])
    hidden = np.zeros((1, hidden_size))
    for i in range(max_len):
        pred, hidden = forward(input, hidden)
        #
        #pred.data *= 10
        temp_dist = softmax(pred)
        temp_dist /= temp_dist.sum()
        pred_y = (temp_dist > np.random.rand()).argmax()
        #pred_y = pred.data.argmax(axis=1)[0]
        #
        s += ind2word[pred_y]
        input = np.array([pred_y])
        
    return s

alpha = 0.001
hidden_size = 512
vocab_len = len(vocab)
batch_size = 32
bptt = 16

word_embed = np.random.normal(0, 0.1, (vocab_len, hidden_size))-0.05
input_layer = np.random.normal(0,0.1, (hidden_size, hidden_size))-0.05
transition_layer = np.eye(hidden_size)
output_layer = np.random.normal(0,0.1, (hidden_size, vocab_len))-0.05
h0 = np.zeros((batch_size, hidden_size))
print('we have words', vocab_len)

epoches = 10
for epoch in range(epoches):
    loss = 0.0
    accuracy = 0.0

    for i in range(n_bptt):
        x = input_batches[i]
        y = target_batches[i]
        
        preds, hidden_state = forward(x, h0)
        cur_loss = cross_entropy_loss_bs(preds, y)
        loss += cur_loss
        accuracy += calc_acc_bs(preds, y)
        input_layer_delta, output_layer_delta, transition_layer_delta, word_embed_delta, h0_delta = backward(preds, hidden_state, x, y)
        step(input_layer_delta, output_layer_delta, transition_layer_delta, word_embed_delta, h0_delta)
    
    print("Perplexity:", np.exp(cur_loss/n_bptt/bptt))
    print('In epoch %d, train loss:%.4f, acc:%.4f' % (epoch, loss/n_bptt/bptt, accuracy/n_bptt/bptt))

init_char = '\n'
predict_byinit_random(word2ind, ind2word, init_char, hidden_size, max_len=1000)