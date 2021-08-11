from .layer import Layer, LinearLayer, Sequential, EmbeddingLayer
from .activation import Sigmoid, Tanh

class RNNCell(Layer):
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
            
        self.parameters = self.input_weights.get_parameters()+self.hidden_state.get_parameters()+\
                           self.activation.get_parameters()+self.output_weights.get_parameters()
        
    def forward(self, input, hidden):
        word_input = self.input_weights.forward(input)
        cur_hidden = self.hidden_state.forward(hidden) + word_input
        cur_hidden = self.activation.forward(cur_hidden)
        output = self.output_weights.forward(cur_hidden)
        
        return output, cur_hidden
    
    def __repr__(self):
        return self.name+':[\n'+super().__repr__()+'\n]\n'

class RNN_Model(Sequential):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(RNN_Model, self).__init__()
        
        self.word_embedding = EmbeddingLayer(vocab_size, embedding_size)
        self.rnn = RNNCell(embedding_size, hidden_size, vocab_size)
        
        self.add(self.word_embedding)
        self.add(self.rnn)
        
    def forward(self, input, hidden):
        word_embeds = self.word_embedding.forward(input)
        output, hidden = self.rnn.forward(word_embeds, hidden)
        
        return output, hidden
        