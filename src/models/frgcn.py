"""FRGCN modules"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv
from models.target_pooling import TragetAttentionPooling
from models.diff_pooling import diff_pooling

class FRGCN(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.

    def __init__(
        self,
        input_dims,
        hidden_dims=32,
        num_layers=4,
        num_relations=3,
        trans_pooling=True
    ):
        super(FRGCN, self).__init__()
        self.convs = th.nn.ModuleList()
        self.convs.append(
            RelGraphConv(
                input_dims,
                hidden_dims,
                num_relations,
                num_bases=2,
                self_loop=True,
            )
        )
        for i in range(num_layers-1):
            self.convs.append(
                RelGraphConv(
                    hidden_dims,
                    hidden_dims,
                    num_relations,
                    num_bases=2,
                    self_loop=True,
                )
            )
        if trans_pooling:
            self.pooling = TragetAttentionPooling(infeat=128, hidden_dim=64)
        else:
            self.pooling = diff_pooling    
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph):
        x = graph.ndata["x"]
        states = []
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(
                conv(
                    graph,
                    x,
                    graph.edata["etype"],
                )
            )
            states.append(x)
        x = th.cat(states, 1)
        x = self.pooling(graph, x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = th.sigmoid(x)
        x = x[:, 0].squeeze()
        return x

    def __repr__(self):
        return self.__class__.__name__
