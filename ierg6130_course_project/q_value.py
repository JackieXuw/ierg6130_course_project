import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn

class QValue(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params_p2 = torch.Tensor(2 * dim).normal_()
        self.params_pp = torch.Tensor(dim, dim, 2).normal_()
        self.params_p2.requires_grad_()
        self.params_pp.requires_grad_()
    
    def forward(self, G, trace_set, v, feature_dict):
        assert isinstance(G, nx.Graph)        
        sum_feature = 0
        sum_feature_num = 0
        for node in trace_set:
            sum_feature += feature_dict[node]
            sum_feature_num += 1
        if sum_feature_num != 0:
            sum_feature = sum_feature / sum_feature_num        
        inp = torch.cat((torch.matmul(self.params_pp[:,:, 0], sum_feature.unsqueeze(1)), torch.matmul(self.params_pp[:,:, 1], \
                        feature_dict[v].unsqueeze(1))), dim=0)

        return torch.matmul(self.params_p2, F.relu(inp))
