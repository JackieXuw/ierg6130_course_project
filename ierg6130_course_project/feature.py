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
from copy import deepcopy

class GraphFeature(nn.Module):

    def __init__(self, config):
        super().__init__()
        graph = config['graph']
        dim = config['feature_dim']
        iteration_radius = config['iteration_radius']
        self.miss_deadline_penalty = config['miss_deadline_penalty'] 
        assert isinstance(graph, nx.DiGraph)
        self.dim = dim
        self.graph = graph
        self.iteration_radius = iteration_radius
        self.params_p = torch.Tensor(dim,6).normal_()
        self.params_p2 = torch.Tensor(dim, 2, 2).normal_()
        self.params_pp = torch.Tensor(dim, dim, 2).normal_()
        self.params_p.requires_grad_()
        self.params_p2.requires_grad_()
        self.params_pp.requires_grad_()
    
    def get_fastest_time(self, node, dest):
        G = self.graph 
        has_path_from_node_to_dest = nx.has_path(G, node, dest)
        if has_path_from_node_to_dest:
            fastest_time = nx.shortest_path(G, node, dest, weight='time')
        else:
            fastest_time = self.miss_deadline_penalty
        return fastest_time
    
    def get_outgoing_min_time_cost_feature(self, node):
        G = self.graph 
        
        adj_cost_feature = 0
        adj_time_feature = 0
        adj_min_cost_edge = {'cost':1e4,
                             'time':0
                             }
        adj_min_time_edge = {'cost':0, 
                             'time':1e4
                            }
        
        for u, adj in G.out_edges(node):
            adj_cost_feature += F.relu(self.params_p[:, 2] * G[node][adj]['cost'])
            adj_time_feature += F.relu(self.params_p[:, 3] * G[node][adj]['time']) 
            if G[node][adj]['cost'] < adj_min_cost_edge['cost']:
                adj_min_cost_edge['cost'] = G[node][adj]['cost']
                adj_min_cost_edge['time'] = G[node][adj]['time']

            if G[node][adj]['time'] < adj_min_time_edge['time']:
                adj_min_time_edge['cost'] = G[node][adj]['cost']
                adj_min_time_edge['time'] = G[node][adj]['time']
        
        adj_min_cost_edge_cost_time = [adj_min_cost_edge['cost'], adj_min_cost_edge['time']]
        adj_min_time_edge_cost_time = [adj_min_time_edge['cost'], adj_min_time_edge['time']] 
        
        return adj_min_cost_edge_cost_time, adj_min_time_edge_cost_time

    def forward(self, state):
        node, remaining_time, dest  = state
        radius = self.iteration_radius 
        G = self.graph 

        # collect the nodes that are within radius-hops of node 
        features_different_slot_list = []
        current_states = {state}
        for current_t in range(radius):
            current_states_feature_dict = dict()
            for current_state in current_states:
                current_nodes_feature_dict[current_state] = None
            features_different_slot_list.append(current_states_feature_dict)
            outgoing_states = {}
            for current_state in current_states:
                v, t_r, d = current_state
                new_states = {(edge[1], t_r - G[edge[0]][edge[1]]['time'], d) for edge in G.out_edges(v)}
                outgoing_states = outgoing_states.union(new_states)

            current_states = deepcopy(outgoing_states) 
        
        # then we start to iterate to get features
        for current_t in range(radius):
            t = radius - (current_t + 1)
            if t == radius - 1:
                # for the farthest loop feature
                current_states = features_different_slot_list[t].keys() 
                for current_state in current_states:
                    u, t_r, d = current_state
                    fastest_time = self.get_fastest_time(u, d) 
                    feature = self.params_p[:, 4] * fastest_time
                    feature += self.params_p[:, 5] * t_r 
                     
                    feature = feature.unsqueeze(1)
                    
                

        

             

        adj_feature = torch.zeros(self.dim)
        adj_feature_num = 0
        adj_cost_feature = 0
        adj_time_feature = 0
                
        # Take average by number of nodes
        if len(list(G.adj)) != 0:
            adj_cost_feature /= len(list(G.adj))
            adj_time_feature /= len(list(G.adj))
        if adj_feature_num != 0:
            adj_feature /= adj_feature_num
        feature = self.params_p[:, 0] * G.nodes[node]['visited']
        
        feature += self.params_p[:, 4] * shortest_path
        feature += self.params_p[:, 5] * remaining_time
        if len(list(G.adj[node])) == 0:
            return feature
        feature += self.params_p[:, 1] * adj_feature
        feature = feature.unsqueeze(1)

        
        
        feature += torch.matmul(self.params_pp[:, :, 0],  adj_cost_feature.unsqueeze(1))
        feature += torch.matmul(self.params_pp[:, :, 1],  adj_time_feature.unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,0], torch.tensor(adj_min_cost_edge_cost_time).unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,1], torch.tensor(adj_min_time_edge_cost_time).unsqueeze(1))
        
        assert feature.shape == (self.dim, 1)
        return feature.squeeze(1)
