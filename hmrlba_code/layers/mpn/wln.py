import torch
from torch import Tensor
from typing import List, Optional, Set
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops

from hmrlba_code.utils import get_global_agg


class WLNConvLast(MessagePassing):

    def __init__(self, hsize: int, bias: bool):
        super(WLNConvLast, self).__init__(aggr='mean')
        self.hsize = hsize
        self.bias = bias
        self._build_components()

    def _build_components(self):
        self.W0 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W2 = nn.Linear(self.hsize, self.hsize, self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        mess = self.W0(x_i) * self.W1(edge_attr) * self.W2(x_j)
        return mess


class WLNConv(MessagePassing):

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 depth: int, hsize: int,
                 bias: bool = False,
                 dropout: float = 0.2,
                 activation: str = 'relu',
                 jk_pool: str = None):
        super(WLNConv, self).__init__(
            aggr='mean')  # We use mean here because the node embeddings started to explode otherwise
        self.hsize = hsize
        self.bias = bias
        self.depth = depth
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.dropout_p = dropout
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'lrelu':
            self.activation_fn = F.leaky_relu
        self.jk_pool = jk_pool
        self._build_components()

    def _build_components(self):
        self.node_emb = nn.Linear(self.node_fdim, self.hsize, self.bias)
        self.mess_emb = nn.Linear(self.edge_fdim, self.hsize, self.bias)

        self.U1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.U2 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.V = nn.Linear(2 * self.hsize, self.hsize, self.bias)

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)
        self.conv_last = WLNConvLast(hsize=self.hsize, bias=self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        if x.size(-1) != self.hsize:
            x = self.node_emb(x)
        edge_attr = self.mess_emb(edge_attr)

        x_depths = []
        for i in range(self.depth):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            x = self.dropouts[i](x)
            x_depth = self.conv_last(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x_depths.append(x_depth)

        x_final = x_depths[-1]
        if self.jk_pool == 'max':
            x_final = torch.stack(x_depths, dim=-1).max(dim=-1)[0]

        elif self.jk_pool == "concat":
            x_final = torch.cat(x_depths, dim=-1)
        return x_final

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        x = self.activation_fn(self.U1(x) + self.U2(inputs))
        return x

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        nei_mess = self.activation_fn(self.V(torch.cat([x_j, edge_attr], dim=-1)))
        return nei_mess


class WLNConvAtt(MessagePassing):

    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 depth: int, hsize: int,
                 bias: bool = False,
                 dropout: float = 0.2,
                 activation: str = 'relu',
                 jk_pool: str = None):
        super(WLNConvAtt, self).__init__(
            aggr='add')  # We use mean here because the node embeddings started to explode otherwise
        self.hsize = hsize
        self.bias = bias
        self.depth = depth
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.dropout_p = dropout
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'lrelu':
            self.activation_fn = F.leaky_relu
        self.jk_pool = jk_pool
        self._build_components()

    def _build_components(self):
        self.node_emb = nn.Linear(self.node_fdim, self.hsize, self.bias)
        self.mess_emb = nn.Linear(self.edge_fdim, self.hsize, self.bias)

        self.U1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.U2 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.V = nn.Linear(2 * self.hsize, self.hsize, self.bias)

        self.a = nn.Parameter(torch.zeros(size=(2 * self.hsize, 1)))
        self.leakrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.a)
        self.lin = nn.Linear(self.hsize, self.hsize, bias=False)
        self.drop_prob = 0

        self.dropouts = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
        self.dropouts = nn.ModuleList(self.dropouts)
        self.conv_last = WLNConvLast(hsize=self.hsize, bias=self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        if x.size(-1) != self.hsize:
            x = self.node_emb(x)
        x = self.lin(x)
        edge_attr = self.mess_emb(edge_attr)

        x_depths = []
        for i in range(self.depth):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            x = self.dropouts[i](x)
            x_depths.append(x)
            x_depth = self.conv_last(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x_depths.append(x_depth)

        x_final = x_depths[-1]
        if self.jk_pool == 'max':
            x_final = torch.stack(x_depths, dim=-1).max(dim=-1)[0]

        elif self.jk_pool == "concat":
            x_final = torch.cat(x_depths, dim=-1)
        return x_final

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        x = self.activation_fn(self.U1(x) + self.U2(inputs))
        return x

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, edge_index_i) -> Tensor:

        e = torch.matmul((torch.cat([x_i, x_j], dim=-1)), self.a)
        e = self.leakrelu(e)
        alpha = softmax(e, edge_index_i)
        alpha = F.dropout(alpha, self.drop_prob, self.training)

        nei_mess = self.activation_fn(self.V(torch.cat([x_j * alpha, edge_attr * alpha], dim=-1)))
        return nei_mess
