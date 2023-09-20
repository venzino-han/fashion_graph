"""Torch modules for graph global pooling."""
import torch as th
import torch.nn as nn

import dgl
from dgl.readout import sum_nodes, softmax_nodes

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

        # self.atte_linear = nn.Linear(hidden_dim, 1)
    
    def _attention_aggregate(self, graph:dgl.DGLGraph, query_vector:th.Tensor, key_vector:th.Tensor, value_vector:th.Tensor):
        scores = th.sum(th.mul(query_vector, key_vector), dim=1).unsqueeze(1)
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

            if get_attention:
                return i_readout, u_readout, i_attentions, u_attentions
            else:
                return i_readout, u_readout


# class TragetAttentionPooling(nn.Module):

#     def __init__(self, infeat=20, hidden_dim=64):
#         super(TragetAttentionPooling, self).__init__()

#         self.query_linear = nn.Linear(infeat, hidden_dim)
#         self.key_linear = nn.Linear(infeat, hidden_dim)
#         self.value_linear = nn.Linear(infeat, hidden_dim)

#         # self.atte_linear = nn.Linear(hidden_dim, 1)

#     def forward(self, graph:dgl.DGLGraph, feat:th.Tensor, target_node_type:int, get_attention:bool=False):
        
#         target_node_feature = feat[graph.ndata['ntype'] == target_node_type].detach()
#         target_feature = th.repeat_interleave(target_node_feature,  graph.batch_num_nodes(), dim=0)
#         with graph.local_scope():
#             # def repeat(input, repeats, dim):
#             #     return th.repeat_interleave(input, repeats, dim)  # PyTorch 1.1

#             target_queries = self.query_linear(target_feature)
#             nighibor_keys = self.key_linear(feat)
#             nighibor_values = self.value_linear(feat)

#             user_feat_center= th.mean(nighibor_values[graph.ndata["ntype"]==2], dim=0)
#             item_feat_center= th.mean(nighibor_values[graph.ndata["ntype"]==3], dim=0)
#             itemset_feat_center= th.mean(nighibor_values[graph.ndata["ntype"]==4], dim=0)

#             ui_dist = th.norm(user_feat_center - item_feat_center)
#             us_dist = th.norm(user_feat_center - itemset_feat_center)
#             si_dist = th.norm(item_feat_center - itemset_feat_center)

#             # print(nighibor_keys.shape) # (batch_nodes, hidden_dim) 
#             scores = th.sum(th.mul(target_queries, nighibor_keys), dim=1).unsqueeze(1)
#             # print(scores.shape) # (batch_nodes, hidden_dim) 
#             graph.ndata['score'] = scores
#             attentions = softmax_nodes(graph, 'score')
#             graph.ndata['r'] = nighibor_values * attentions
#             readout = sum_nodes(graph, 'r')
#             graph.ndata.pop('r')

#             if get_attention:
#                 return readout, -ui_dist-us_dist-si_dist, attentions
#             else:
#                 return readout, -ui_dist-us_dist-si_dist



# class TragetAttentionPooling(nn.Module):

#     def __init__(self, gate_nn, feat_nn=None):
#         super(TragetAttentionPooling, self).__init__()
#         self.gate_nn = gate_nn
#         self.feat_nn = feat_nn

#     def forward(self, graph:dgl.DGLGraph, feat:th.Tensor, target_node_type:int, get_attention:bool=False):
#         r"""

#         Compute global attention pooling.

#         Parameters
#         ----------
#         graph : DGLGraph
#             A DGLGraph or a batch of DGLGraphs.
#         feat : torch.Tensor
#             The input node feature with shape :math:`(N, D)` where :math:`N` is the
#             number of nodes in the graph, and :math:`D` means the size of features.
#         get_attention : bool, optional
#             Whether to return the attention values from gate_nn. Default to False.

#         Returns
#         -------
#         torch.Tensor
#             The output feature with shape :math:`(B, D)`, where :math:`B` refers
#             to the batch size.
#         torch.Tensor, optional
#             The attention values of shape :math:`(N, 1)`, where :math:`N` is the number of
#             nodes in the graph. This is returned only when :attr:`get_attention` is ``True``.
#         """
#         with graph.local_scope():
#             #get target node feature from graph
#             target_node_feature = feat[graph.ndata['ntype'] == target_node_type]
#             # print(target_node_feature.shape)
#             # print(feat.shape)
                        
#             # graph.ndata['target_feature'] = broadcast_nodes(graph, target_node_feature)
#             target_feature = broadcast_nodes(graph, target_node_feature)

#             new_feat = th.cat([feat, target_feature], dim=-1)
#             gate = self.gate_nn(new_feat)
#             assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
#             feat = self.feat_nn(new_feat) if self.feat_nn else new_feat


#             graph.ndata['gate'] = gate
#             gate = softmax_nodes(graph, 'gate')
#             graph.ndata.pop('gate')

#             graph.ndata['r'] = feat * gate
#             readout = sum_nodes(graph, 'r')
#             graph.ndata.pop('r')

#             if get_attention:
#                 return readout, gate
#             else:
#                 return readout

