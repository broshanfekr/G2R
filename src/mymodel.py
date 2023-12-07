import torch
from torch import nn
import torch.nn.functional as F

class GCNConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 has_bias):
        super(GCNConv, self).__init__()
        
        self.input_dim = in_channels
        self.output_dim = out_channels
        self.has_bias = has_bias
        self.dropout = nn.Dropout(p=dropout)

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

        # Convolution Operation
        pre_sup = torch.mm(x, self.weight)
        output = torch.mm(adj, pre_sup)

        if self.has_bias:
            output += self.bias

        return output
    

class Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 activation,
                 base_model,
                 dropout: float,
                 k: int = 2,
                 has_bias=True):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.k = k

        self.conv_0 = self.base_model(in_channels,  out_channels, dropout=dropout, has_bias=has_bias)
        self.conv_1 = self.base_model(in_channels,  hidden_channels, dropout=dropout, has_bias=has_bias)
        self.conv_2 = self.base_model(hidden_channels, hidden_channels, dropout=dropout, has_bias=has_bias)
        self.conv_3 = self.base_model(hidden_channels, hidden_channels, dropout=dropout, has_bias=has_bias)

        self.conv_last_layer = self.base_model(hidden_channels, out_channels, dropout=dropout, has_bias=has_bias)
        self.conv_layers_list = [self.conv_1, self.conv_2, self.conv_3]
        self.activation = activation
        self.prelu = nn.PReLU(out_channels)
        self.dropout = nn.Dropout(p=dropout)

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
