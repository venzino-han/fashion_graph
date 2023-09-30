"""IGMC modules"""
import numpy as np
import torch as th
import torch.nn as nn
from dgl.nn.pytorch.conv import EGATConv
from models.target_pooling import TragetAttentionPooling


class FGAT(nn.Module):
    def __init__(
        self,
        in_nfeats,
        in_efeats,
        latent_dim,
        num_heads=2,
        edge_dropout=0.2,
    ):
        super(FGAT, self).__init__()
        self.edge_dropout = edge_dropout
        self.in_nfeats = in_nfeats
        self.elu = nn.ELU()
        self.leakyrelu = th.nn.LeakyReLU()
        self.convs = th.nn.ModuleList()


        self.convs.append(
            EGATConv(
                in_node_feats=in_nfeats,
                out_node_feats=latent_dim[0],
                in_edge_feats=in_efeats,
                out_edge_feats=in_efeats,
                num_heads=num_heads,
            )
        )

        for i in range(0, len(latent_dim) - 1):
            self.convs.append(
                EGATConv(
                    in_node_feats=latent_dim[i],
                    out_node_feats=latent_dim[i+1],
                    in_edge_feats=in_efeats,
                    out_edge_feats=in_efeats,
                    num_heads=num_heads,
                )
            )

        self.pooling = TragetAttentionPooling(infeat=128, hidden_dim=64)  # create a Global Attention Pooling layer

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
        """graph : subgraph"""
        # graph = edge_drop(graph, self.edge_dropout,)

        # graph.edata["norm"] = graph.edata["edge_mask"]
        x = graph.ndata["x"].float()
        e = graph.edata["efeat"]

        states = []
        for conv in self.convs:
            x, _ = conv(graph=graph, nfeats=x, efeats=e)
            x = th.sum(x, dim=1)
            x = self.elu(x)
            states.append(x)

        x = th.cat(states, 1)
        x_u, x_i, intra_cl_loss = self.pooling(graph, x)
        x = th.cat([x_u, x_i], 1)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)
        x = x[:, 0].squeeze()
        return x, intra_cl_loss
