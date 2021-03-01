import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def activation(input, type):
    if type.lower() == 'selu':
        return F.selu(input)
    elif type.lower() == 'elu':
        return F.elu(input)
    elif type.lower() == 'relu':
        return F.relu(input)
    elif type.lower() == 'relu6':
        return F.relu6(input)
    elif type.lower() == 'tanh':
        return F.tanh(input)
    elif type.lower() == 'sigmoid':
        return F.sigmoid(input)
    elif type.lower() == 'swish':
        return F.sigmoid(input) * input
    elif type.lower() == 'identity':
        return input
    else:
        raise ValueError("Unknown non-Linearity activation function")


class AutoEncoder(nn.Module):
    def __init__(self, layer_size, nl_type='selu', is_constrained=True, dp_drop_prob= 0, last_layer_activations=True):
        super(AutoEncoder, self).__init__()
        '''
        layer_sizes: size of each layer in the autoencoder model
            ex) [10000, 1024, 512] will result in
                - encoder 2 layers: 10000 x 1024 & 1024 x 512
                - representation layer z: 512
                - decoder 2 layers: 512 x 1024 & 1024 x 10000
        nl_type: non-linearity type
        is_constrained: if ture then the weights of encoder and decoder are tied
        dp_drop_prob: dropout probability
        last_layer_activations: whether to apply activation on last decoder layer
        '''

        self.layer_sizes= layer_size
        self.nl_type= nl_type
        self.is_constrained= is_constrained
        self.dp_drop_prob= dp_drop_prob
        self.last_layer_activations= last_layer_activations

        if dp_drop_prob > 0:
            self.drop= nn.Dropout(dp_drop_prob)

        self._last= len(layer_size) - 2

        # initialize weights
        self.encoder_weights= nn.ParameterList([nn.Parameter(torch.rand(layer_size[i+1], layer_size[i])) for i in range(len(layer_size)-1)])

        for weights in self.encoder_weights:
            init.xavier_uniform_(weights)

        self.encoder_bias= nn.ParameterList([nn.Parameter(torch.zeros(layer_size[i+1])) for i in range(len(layer_size) - 1)])

        reverse_layer_sizes= list(reversed(layer_size))

        # Decoder weights
        if is_constrained == False:
            self.decoder_weights= nn.ParameterList([nn.Parameter(torch.rand(reverse_layer_sizes[i+1], reverse_layer_sizes[i])) for i in range(len(reverse_layer_sizes) - 1)])

            for weights in self.decoder_weights:
                init.xavier_uniform_(weights)

        self.decoder_bias= nn.ParameterList([nn.Parameter(torch.zeros(reverse_layer_sizes[i+1])) for i in range(len(reverse_layer_sizes) - 1)])

    def encode(self, x):
        for i, w in enumerate(self.encoder_weights):
            x= F.linear(input=x, weight=w, bias=self.encoder_bias[i])
            x= activation(input=x, type=self.nl_type)

        if self.dp_drop_prob > 0:
            x= self.drop(x)

        return x

    def decode(self, x):
        if self.is_constrained == True:
            for i, w in zip(range(len(self.encoder_weights)), list(reversed(self.encoder_weights))):
                x= F.linear(input=x, weight=w.t(), bias=self.decoder_bias[i])
                x= activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'identity')

        else:
            for i, w in enumerate(self.decoder_weights):
                x= F.linear(input=x, weight=w, bias=self.decoder_bias[i])
                x= activation(input=x, type=self.nl_type if i != self._last or self.last_layer_activations else 'identity')

        return x

    def forward(self, x):
        return self.decode(self.encode(x))
