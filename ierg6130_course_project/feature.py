"""
We construct features to process the state (current node, destination, remaining time) for further training purpose.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from copy import deepcopy
from queue import Queue
from torch.optim import Adam
import argparse
import time
import os
import logging

class HandcraftedFeature(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params_p = torch.Tensor(dim,6).normal_()
        self.params_p2 = torch.Tensor(dim, 2, 2).normal_()
        self.params_pp = torch.Tensor(dim, dim, 2).normal_()
        self.params_p.requires_grad_()
        self.params_p2.requires_grad_()
        self.params_pp.requires_grad_()

    def forward(self, G, node, dest, rTime, visited = True):
        if G.nodes[node]['dest'] != dest:
            G.nodes[node]['visited'] = 0
            G.nodes[node]['dest'] = dest
            G.nodes[node]['feature'] = dict()
        adj_feature = torch.zeros(self.dim)
        adj_feature_num = 0
        adj_cost_feature = 0
        adj_time_feature = 0
        adj_min_cost = [1e4, 0]
        adj_min_time = [0, 1e4]
        try:
            # whether can we go from node to dest
            shortest_path = nx.shortest_path_length(G, node, dest, 'time')
        except:
            shortest_path = 1e2 # Very large punishment
        
        for adj in G.adj[node]:
            adj_cost_feature += F.relu(self.params_p[:, 2] * G[node][adj]['cost'])
            adj_time_feature += F.relu(self.params_p[:, 3] * G[node][adj]['time'])
            if G[node][adj]['cost'] < adj_min_cost[0]:
                adj_min_cost[0] = G[node][adj]['cost']
                adj_min_cost[1] = G[node][adj]['time']
            if G[node][adj]['time'] < adj_min_time[1]:
                adj_min_time[0] = G[node][adj]['cost']
                adj_min_time[1] = G[node][adj]['time']
            if G.nodes[adj]['dest'] == dest and '{:.3f}'.format(rTime - G[node][adj]['time']) in G.nodes[adj]['feature']:
                adj_feature += G.nodes[adj]['feature']['{:.3f}'.format(rTime - G[node][adj]['time'])]
                adj_feature_num += 1
        # Take average by number of nodes
# <<<<<<< HEAD
        if len(list(G.adj)) != 0:
            adj_cost_feature /= len(list(G.adj))
            adj_time_feature /= len(list(G.adj))
        if adj_feature_num != 0:
            adj_feature /= adj_feature_num
        feature = self.params_p[:, 0] * G.nodes[node]['visited']
        if not visited:
            feature = torch.zeros(feature.shape)
# =======
#         adj_cost_feature /= len(list(G.adj))
#         adj_time_feature /= len(list(G.adj))
#         adj_feature /= adj_feature_num
#         feature = self.params_p[:, 0] * G.nodes[node]['visited']
# >>>>>>> patch-1
        feature += self.params_p[:, 4] * shortest_path
        feature += self.params_p[:, 5] * rTime
        if len(list(G.adj[node])) == 0:
            return feature
        feature += self.params_p[:, 1] * adj_feature
        feature = feature.unsqueeze(1)
        
        feature += torch.matmul(self.params_pp[:, :, 0],  adj_cost_feature.unsqueeze(1))
        feature += torch.matmul(self.params_pp[:, :, 1],  adj_time_feature.unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,0], torch.tensor(adj_min_cost).unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,1], torch.tensor(adj_min_time).unsqueeze(1))
        assert feature.shape == (self.dim, 1)
        return feature.squeeze(1)
