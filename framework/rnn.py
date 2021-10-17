from .layer import Layer, Sequential
from .parameter import Parameter
from .linear import LinearLayer
from .activation import Sigmoid, Tanh
import numpy as np


class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingLayer, self).__init__()
        w = (np.random.rand(vocab_size, hidden_size)-0.5)/hidden_size
        self.embedding_weights = Parameter(self.get_name('Embedding_Weights_'), w, requires_grad=True)

    def forward(self, words):
        return self.embedding_weights.index_select(words)

class RNNCell(Sequential):
    def __init__(self, embedding_size, hidden_size, vocab_size, activation='sigmoid'):
        super(RNNCell, self).__init__()
        self.name = self.get_name('RNNCell_')

        self.input_weights = LinearLayer(embedding_size, hidden_size)
        self.hidden_state = LinearLayer(hidden_size, hidden_size)
        self.output_weights = LinearLayer(hidden_size, vocab_size)
        
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            self.activation = Tanh()
        
    def forward(self, input, hidden):
        word_input = self.input_weights.forward(input)
        cur_hidden = self.hidden_state.forward(hidden) + word_input
        cur_hidden = self.activation.forward(cur_hidden)
        output = self.output_weights.forward(cur_hidden)
        
        return output, cur_hidden
    
    #def __repr__(self):
    #    return self.name+':[\n'+super().__repr__()+'\n]\n'

class LstmCell(Sequential):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(LstmCell, self).__init__()
        self.name = self.get_name('LstmCell_')

        # forget gate
        self.forget_weights_h = LinearLayer(hidden_size, hidden_size)
        self.forget_weights_i = LinearLayer(embedding_size, hidden_size)

        # input gate
        self.input_weights_h = LinearLayer(hidden_size, hidden_size)
        self.input_weights_i = LinearLayer(embedding_size, hidden_size)

        # output gate
        self.output_weights_h = LinearLayer(hidden_size, hidden_size)
        self.output_weights_i = LinearLayer(embedding_size, hidden_size)
        
        # update
        self.update_weights_h = LinearLayer(hidden_size, hidden_size)
        self.update_weights_i = LinearLayer(embedding_size, hidden_size)
        
        # output 
        self.output = LinearLayer(hidden_size, output_size)

        # activation
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, input, hidden):
        prev_hidden, prev_c = hidden

        f = self.sigmoid(self.forget_weights_h.forward(prev_hidden) + self.forget_weights_i.forward(input))
        i = self.sigmoid(self.input_weights_h.forward(prev_hidden) + self.input_weights_i.forward(input))
        o = self.sigmoid(self.output_weights_h.forward(prev_hidden) + self.output_weights_i.forward(input))
        u = self.tanh(self.update_weights_h.forward(prev_hidden) + self.update_weights_i.forward(input))
        
        cell_state = f*prev_c + i*u
        hidden_state = o*cell_state.tanh()
        output = self.output.forward(hidden_state)
        
        return output, (hidden_state, cell_state)
    
    #def __repr__(self):
    #    return self.name+':[\n'+super().__repr__()+'\n]\n'

class RNN_Model(Sequential):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(RNN_Model, self).__init__()
        self.name = self.__class__.__name__

        self.word_embedding = EmbeddingLayer(vocab_size, embedding_size)
        self.rnn = RNNCell(embedding_size, hidden_size, vocab_size)
        
    def forward(self, input, hidden):
        word_embeds = self.word_embedding.forward(input)
        output, hidden = self.rnn.forward(word_embeds, hidden)
        
        return output, hidden

class Lstm_Model(Sequential):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(Lstm_Model, self).__init__()
        self.name = self.__class__.__name__

        self.word_embedding = EmbeddingLayer(vocab_size, embedding_size)
        self.lstm = LstmCell(embedding_size, hidden_size, vocab_size)
        
    def forward(self, input, hidden):
        word_embeds = self.word_embedding.forward(input)
        output, hidden = self.lstm.forward(word_embeds, hidden)
        
        return output, hidden