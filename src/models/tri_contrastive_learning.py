# module for triple contrastive learning

import dgl
from dgl.readout import sum_nodes, mean_nodes, softmax_nodes, broadcast_nodes

import torch as th
import torch.nn as nn

class TripletContrastiveLoss(nn.Module):
    def __init__(self, method="center"):
        super(TripletContrastiveLoss, self).__init__()
        self.method = method

    def forward(self, graph:dgl.DGLGraph, feat:th.Tensor):
        if self.method == "center":
            # get user nodes
            user_feat_center= th.mean(feat[graph.ndata["ntype"]==2])

            # get item nodes
            item_feat_center= th.mean(feat[graph.ndata["ntype"]==3])

            # get itemset nodes
            itemset_feat_center= th.mean(feat[graph.ndata["ntype"]==4])

            ui_dist = th.norm(user_feat_center - item_feat_center)
            us_dist = th.norm(user_feat_center - itemset_feat_center)
            si_dist = th.norm(item_feat_center - itemset_feat_center)
            return -ui_dist-us_dist-si_dist




        
        


