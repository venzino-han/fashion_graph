"""FRGCN modules"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.LightGCN import LGCNLayer
from dgl.nn.pytorch.glob import SetTransformerEncoder
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from models.target_pooling import TragetAttentionPooling
from models.tri_contrastive_learning import TripletContrastiveLoss

class FLGCN(nn.Module):

    def __init__(
        self,
        in_feats,
        latent_dim=[32, 32, 32, 32],
        edge_dropout=0.2,
    ):
        super(FLGCN, self).__init__()
        self.convs = th.nn.ModuleList()
        self.convs.append(LGCNLayer())
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(LGCNLayer())

        # self.pooling = TransPooling(in_feats=in_feats, hidden_dim=32)
        # self.pooling = GlobalAttentionPooling(gate_nn)  # create a Global Attention Pooling layer
        # gate_nn = th.nn.Linear(40, 1)  # the gate layer that maps node feature to scalar
        # feat_nn = th.nn.Linear(40, 64)  # the gate layer that maps node feature to scalar
        # self.pooling_u = TragetAttentionPooling(gate_nn, feat_nn)  # create a Global Attention Pooling layer
        # gate_nn2 = th.nn.Linear(40, 1)  # the gate layer that maps node feature to scalar
        # feat_nn2 = th.nn.Linear(40, 64)  # the gate layer that maps node feature to scalar
        # self.pooling_i = TragetAttentionPooling(gate_nn2, feat_nn2)  # create a Global Attention Pooling layer

        self.pooling = TragetAttentionPooling()  # create a Global Attention Pooling layer
        # self.pooling_i = TragetAttentionPooling()  # create a Global Attention Pooling layer

        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 1)
        self.reset_parameters()

        # self.triplet_loss = TripletContrastiveLoss()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph):
        concat_states = []
        x = graph.ndata["x"].type(
            th.float32
        )  # one hot feature to emb vector : this part fix errors

        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = conv(graph,x)
            concat_states.append(x)
        x = th.cat(concat_states, 1)

        x_u, x_i = self.pooling(graph, x)
        # x_i, tri_loss_i = self.pooling_i(graph, x, target_node_type=1)

        x = th.cat([x_u, x_i], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = th.sigmoid(x)
        x = x[:, 0].squeeze()
        return x, None

    def __repr__(self):
        return self.__class__.__name__

