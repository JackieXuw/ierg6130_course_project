import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from feature import GraphFeature

class GraphFeatureQValue(nn.Module):
    def __init__(self, dim, graph):
        super().__init__()
        assert isinstance(graph, nx.Graph)
        self.dim = dim
        self.params_p2 = torch.Tensor(2 * dim).normal_()
        self.params_pp = torch.Tensor(dim, dim, 2).normal_()
        self.params_p2.requires_grad_()
        self.params_pp.requires_grad_()
        self.graph_feature = GraphFeature(dim, graph, iteration_radius) 

    def forward(self, state, action):
        """
        Get the approximated Q-Value using the graph feature. 
        """
        state_feature = self.graph_feature(state)
        action_feature = self.graph_feature(action) 
        inp = torch.cat((torch.matmul(self.params_pp[:,:, 0], state_feature.unsqueeze(1)), torch.matmul(self.params_pp[:,:, 1], \
                        action_feature.unsqueeze(1))), dim=0)

        return torch.matmul(self.params_p2, F.relu(inp))
