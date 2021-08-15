from framework.tensor import Tensor
from framework.rnn import RNN_Model, Lstm_Model
import numpy as np

# print tensor 
a = Tensor(np.array([1,2,3]))
print(a)

# print model
word_embedding_size = 64
hidden_size = 64
vocab_size = 1024
model = RNN_Model(word_embedding_size, hidden_size, vocab_size)
print(model)

model = Lstm_Model(word_embedding_size, hidden_size, vocab_size)
print(model)