"""FLGCN modules"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.LightGCN import LGCNLayer
from models.target_pooling import TragetAttentionPooling
from models.diff_pooling import diff_pooling

class FLGCN(nn.Module):

    def __init__(
        self,
        input_dims,
        hidden_dims=32,
        num_layers=4,
        trans_pooling=True
    ):
        super(FLGCN, self).__init__()
        self.convs = th.nn.ModuleList()
        self.convs.append(LGCNLayer())
        for i in range(num_layers-1):
            self.convs.append(LGCNLayer())
        if trans_pooling:
            self.pooling = TragetAttentionPooling()  # create a Global Attention Pooling layer
            self.lin1 = nn.Linear(hidden_dims*4, 64)
        else: 
            self.pooling = diff_pooling
            self.lin1 = nn.Linear(input_dims*4, 64)
        self.lin2 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph):
        x = graph.ndata["x"] 
        states = []
        for conv in self.convs:
            x = conv(graph,x)
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

