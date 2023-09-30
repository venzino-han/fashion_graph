"""FGAT modules"""
import numpy as np
import torch as th
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
from models.target_pooling import TragetAttentionPooling
from models.diff_pooling import diff_pooling

class FGAT(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_dims=32,
        num_layers=4,
        num_heads=2,
        trans_pooling=True
    ):
        super(FGAT, self).__init__()
        self.elu = nn.ELU()
        self.leakyrelu = th.nn.LeakyReLU()
        self.convs = th.nn.ModuleList()
        self.convs.append(
            GATConv(
                in_feats=input_dims,
                out_feats=hidden_dims,
                num_heads=num_heads,
            )
        )
        for i in range(num_layers-1):
            self.convs.append(
                GATConv(
                    in_feats=hidden_dims,
                    out_feats=hidden_dims,
                    num_heads=num_heads,
                )
            )
        if trans_pooling:
            self.pooling = TragetAttentionPooling(infeat=128, hidden_dim=64)  # create a Global Attention Pooling layer
        else:
            self.pooling = diff_pooling
        self.lin1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_parameters(self):
        parameters_dict = {}
        n, p = self.lin1.named_parameters()
        parameters_dict[n] = p
        n, p = self.lin2.named_parameters()
        parameters_dict[n] = p
        return parameters_dict

    def forward(self, graph):
        x = graph.ndata["x"]
        states = []
        for conv in self.convs:
            x = conv(graph=graph, feat=x)
            x = th.sum(x, dim=1)
            x = self.elu(x)
            states.append(x)
        x = th.cat(states, 1)
        x = self.pooling(graph, x)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)
        x = x[:, 0].squeeze()
        return x
