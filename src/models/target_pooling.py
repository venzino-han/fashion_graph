"""Torch modules for graph global pooling."""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import dgl
from dgl.readout import sum_nodes, softmax_nodes
from dgl.nn.pytorch.glob import AvgPooling

from typing import Tuple

class TragetAttentionPooling(nn.Module):

    def __init__(self, infeat=20, hidden_dim=64):
        super(TragetAttentionPooling, self).__init__()

        self.i_query_linear = nn.Linear(infeat, hidden_dim)
        self.i_key_linear = nn.Linear(infeat, hidden_dim)
        self.i_value_linear = nn.Linear(infeat, hidden_dim)
        
        self.u_query_linear = nn.Linear(infeat, hidden_dim)
        self.u_key_linear = nn.Linear(infeat, hidden_dim)
        self.u_value_linear = nn.Linear(infeat, hidden_dim)
    
    def _attention_aggregate(self, graph:dgl.DGLGraph, query_vector:th.Tensor, key_vector:th.Tensor, value_vector:th.Tensor):
        scores = th.sum(th.mul(query_vector, key_vector), dim=1).unsqueeze(1)/np.sqrt(key_vector.shape[-1])
        graph.ndata['score'] = scores
        attentions = softmax_nodes(graph, 'score')
        graph.ndata['r'] = value_vector * attentions
        readout = sum_nodes(graph, 'r')
        graph.ndata.pop('r')
        return readout, attentions

    def forward(self, graph:dgl.DGLGraph, feat:th.Tensor, target_node_types:Tuple[int]=(0,1), get_attention:bool=False):
        ntype_i = target_node_types[0]
        ntype_u = target_node_types[1]
        i_node_feature = feat[graph.ndata['ntype'] == ntype_i].detach()
        u_node_feature = feat[graph.ndata['ntype'] == ntype_u].detach()
        i_target_feature = th.repeat_interleave(i_node_feature,  graph.batch_num_nodes(), dim=0)
        u_target_feature = th.repeat_interleave(u_node_feature,  graph.batch_num_nodes(), dim=0)
        
        with graph.local_scope():

            i_target_queries = self.i_query_linear(i_target_feature)
            u_target_queries = self.u_query_linear(u_target_feature)
            
            i_nighibor_keys = self.i_key_linear(feat)
            i_nighibor_values = self.i_value_linear(feat)
            u_nighibor_keys = self.u_key_linear(feat)
            u_nighibor_values = self.u_value_linear(feat)

            i_readout, i_attentions = self._attention_aggregate(graph, i_target_queries, i_nighibor_keys, i_nighibor_values)
            u_readout, u_attentions = self._attention_aggregate(graph, u_target_queries, u_nighibor_keys, u_nighibor_values)

            readout = th.cat([i_readout, u_readout], 1)

            if get_attention:
                return readout, i_attentions, u_attentions
            else:
                return readout
