import torch
from torch import nn
import torch.nn.functional as F


class GraphLearn(nn.Module):
    def __init__(self, input_dim, output_dim, edges, num_nodes, bias, device, dropout=0., act=nn.ReLU()) -> None:
        super(GraphLearn, self).__init__()

        self.edges = edges
        self.num_nodes = num_nodes
        self.act = act
        self.bias = bias
        self.device = device

        self.w = nn.Linear(input_dim, output_dim, bias=self.bias)
        self.a = nn.Linear(output_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.eye = torch.eye(num_nodes).to(self.device)

    def forward(self, inputs):
        x = self.dropout(inputs)

        # graph learning
        h = self.w(x)
        N = self.num_nodes
        first = torch.index_select(h, 0, self.edges[0])
        sec = torch.index_select(h, 0, self.edges[1])

        edge_v = torch.abs(first - sec)
        edge_v = torch.squeeze(self.act(self.a(edge_v)))

        graph = torch.sparse_coo_tensor(indices=self.edges, values=edge_v, size=[N, N])
        graph = torch.sparse.softmax(graph, dim=1).to_dense()
        # graph = graph + self.eye

        graph = (graph + graph.T)/2
        graph = graph + self.eye
        D = torch.sum(graph, dim=1)
        D = torch.sqrt(D)
        D = torch.diag(D)
        D = torch.pinverse(D)
        graph = torch.mm(torch.mm(D, graph), D)

        return h, graph


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
        return x, x, A
    

class MyEncoder(torch.nn.Module):
    def __init__(self, 
                 edges, 
                 num_nodes, 
                 input_dim, 
                 output_dim, 
                 hidden_gl_dim, 
                 hidden_gcn_dim, 
                 dropout,
                 activation, 
                 device,
                 has_bias=True):
        super().__init__()

        self.edge = edges
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_gl_dim = hidden_gl_dim
        self.hidden_gcn_dim = hidden_gcn_dim
        self.has_bias = has_bias
        self.dropout = dropout
        self.activation = activation
        self.prelu = nn.PReLU(output_dim)
        self.device = device

        self.layers0 = GraphLearn(
            input_dim=self.input_dim,
            output_dim=self.hidden_gl_dim, 
            edges=edges, 
            num_nodes=self.num_nodes,
            device=self.device,
            bias=self.has_bias,
            dropout=self.dropout,
            act=nn.ReLU()
        )

        self.layer1 = GCNConv(in_channels=self.input_dim,  
                              out_channels=self.hidden_gcn_dim, 
                              dropout=self.dropout, 
                              has_bias=self.has_bias
        )

        self.layer2 = GCNConv(in_channels=self.hidden_gcn_dim, 
                              out_channels=self.output_dim,
                              dropout=self.dropout,
                              has_bias=has_bias
        )


    def forward(self, x: torch.Tensor, A: torch.Tensor):
        h, A_hat = self.layers0(x)
        x = self.activation(self.layer1(x, A_hat))
        x = self.layer2(x, A_hat)
        x = F.normalize(x, p=1)
        return x, h, A_hat