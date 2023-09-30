""" diff pooling layer """

import dgl
import torch as th

def diff_pooling(graph:dgl.DGLGraph, x:th.Tensor) -> th.Tensor:
    """ 
    diff pooling layer 
    select two target node embeddings and concat them
    """
    u = graph.ndata["ntype"] == 1
    v = graph.ndata["ntype"] == 0
    x = x[u] + x[v]
    return x

