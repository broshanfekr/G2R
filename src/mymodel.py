import torch
from torch import nn
import torch.nn.functional as F


class GCNConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 has_bias=True,
                 normalize=True,
                 dropout=0.3,
                 **kwargs):
        super(GCNConv, self).__init__(**kwargs)
        
        self.input_dim = in_channels
        self.output_dim = out_channels
        self.has_bias = has_bias
        self.dropout = nn.Dropout(p=dropout)
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        if has_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.has_bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs, adj):
        x = inputs
        x = self.dropout(inputs)

        # convolve
        pre_sup = torch.mm(x, self.weight)
        output = torch.mm(adj, pre_sup)

        if self.has_bias:
            output += self.bias

        return output
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.input_dim,
                                   self.output_dim)
    

class Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 activation,
                 base_model=GCNConv,
                 k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.k = k

        self.conv_0 = self.base_model(in_channels,  out_channels)
        self.conv_1 = self.base_model(in_channels,  hidden_channels)
        self.conv_2 = self.base_model(hidden_channels, hidden_channels)
        self.conv_3 = self.base_model(hidden_channels, hidden_channels)

        self.conv_last_layer = self.base_model(hidden_channels, out_channels)
        self.conv_layers_list = [self.conv_1, self.conv_2, self.conv_3]
        self.activation = activation
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        if self.k == 0:
            x = self.conv_0(x, A)
            x = F.normalize(x, p=1)
            return x
        for i in range(0, self.k):
            x = self.activation(self.conv_layers_list[i](x, A))
        x = self.conv_last_layer(x, A)
        x = F.normalize(x, p=1)
        return x
