import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from feature import GraphFeature

class GraphFeatureQValue(nn.Module):
    def __init__(self, graph, config):
        super().__init__()
        assert isinstance(graph, nx.Graph)
        dim = config['feature_dim']
        iter_radius = config['iteration_radius']
        params_init_scale = config['params_init_scale'] 
        self.dim = dim
        self.params_p2 = torch.nn.Parameter(params_init_scale *
                                            torch.Tensor(2 * dim).normal_(),
                                            requires_grad=True)
        self.params_pp = torch.nn.Parameter(params_init_scale *
                                            torch.Tensor(dim, dim, 2).normal_(),
                                            requires_grad=True)
        self.graph_feature = GraphFeature(config)

    def forward(self, state, action):
        """
        Get the approximated Q-Value using the graph feature. 
        """
        state_feature = self.graph_feature(state)
        action_feature = self.graph_feature(action) 
        inp = torch.cat((torch.matmul(self.params_pp[:,:, 0], state_feature.unsqueeze(1)), torch.matmul(self.params_pp[:,:, 1], \
                        action_feature.unsqueeze(1))), dim=0)

        return torch.matmul(self.params_p2, inp)
