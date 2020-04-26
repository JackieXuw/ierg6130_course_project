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
        params_init_scale = config['params_init_scale'] 
        self.miss_deadline_penalty = config['miss_deadline_penalty']
        self.time_radius = config['time_radius']
        self.cost_radius = config['cost_radius'] 
        if 'node2node_time' and 'node2node_cost' in config.keys():
            self.node2node_time = config['node2node_time']
            self.node2node_cost = config['node2node_cost'] 
        assert isinstance(graph, nx.DiGraph)
        self.dim = dim
        self.graph = graph
        self.iteration_radius = iteration_radius
        self.params_p = torch.nn.Parameter(params_init_scale *
                                           torch.Tensor(dim, 6).normal_(),
                                           requires_grad=True)
        self.params_p2 = torch.nn.Parameter(params_init_scale *
                                            torch.Tensor(dim, 2, 2).normal_(),
                                            requires_grad=True)
        self.params_pp = torch.nn.Parameter(params_init_scale *
                                            torch.Tensor(dim, dim, 2).normal_(),
                                            requires_grad=True)


    def get_fastest_time(self, node, dest):
        try:
            fastest_time = self.node2node_time[(node, dest)]
            return fastest_time
        except Exception as e:
            pass
        G = self.graph 
        #import pdb; pdb.set_trace()
        has_path_from_node_to_dest = nx.has_path(G, node, dest)
        if has_path_from_node_to_dest:
            fastest_time = nx.shortest_path_length(G, node, dest, weight='time'
                                                   )
        else:
            fastest_time = self.time_radius
        return fastest_time
    
    def add_incident_edge_feature(self, node, feature):
        G = self.graph 
        
        adj_cost_feature = torch.zeros(self.dim)
        adj_time_feature = torch.zeros(self.dim)
        adj_min_cost_edge = {'cost': 1e4,
                             'time': 0
                             }
        adj_min_time_edge = {'cost': 0, 
                             'time': 1e4
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

        num_out_edges = len(list(G.out_edges(node)))
        if num_out_edges != 0:
            adj_cost_feature /= num_out_edges 
            adj_time_feature /= num_out_edges 
        
        adj_min_cost_edge_cost_time = [adj_min_cost_edge['cost'], adj_min_cost_edge['time']]
        adj_min_time_edge_cost_time = [adj_min_time_edge['cost'], adj_min_time_edge['time']] 
                
        feature += torch.matmul(self.params_pp[:, :, 0],  adj_cost_feature.unsqueeze(1))
        feature += torch.matmul(self.params_pp[:, :, 1],  adj_time_feature.unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,0], torch.tensor(adj_min_cost_edge_cost_time).unsqueeze(1))
        feature += torch.matmul(self.params_p2[:,:,1], torch.tensor(adj_min_time_edge_cost_time).unsqueeze(1))
        
        return feature 

        
    def forward(self, state):
        node, dest, remaining_time = state
        radius = self.iteration_radius 
        G = self.graph 

        # collect the nodes that are within radius-hops of node 
        features_different_slot_list = []
        current_states = {state}
        states_to_new_states = dict()
        for current_t in range(radius):
            current_states_feature_dict = dict()
            for current_state in current_states:
                current_states_feature_dict[current_state] = None
            features_different_slot_list.append(current_states_feature_dict)
            outgoing_states = set()
            for current_state in current_states:
                v, d, t_r = current_state
                new_states = {(edge[1], d, t_r - G[edge[0]][edge[1]]['time']) \
                    for edge in G.out_edges(v)}
                states_to_new_states[current_state] = deepcopy(new_states) 
                outgoing_states = outgoing_states.union(new_states)

            current_states = deepcopy(outgoing_states) 
        
        # then we start to iterate to get features
        for t in reversed(range(radius)):
            current_states = features_different_slot_list[t].keys() 
            for current_state in current_states:
                u, d, t_r = current_state
                u = int(u)
                d = int(d)
                # add time feature
                fastest_time = self.get_fastest_time(u, d) 
                feature = self.params_p[:, 4] * fastest_time
                feature += self.params_p[:, 5] * t_r 

                feature = feature.unsqueeze(1)
                # add average of adj features 
                if t < radius - 1:
                    adj_feature = torch.zeros(self.dim, 1)
                    out_states = states_to_new_states[current_state]
                    adj_feature_num = len(out_states)
                    for state in out_states:
                        adj_feature += features_different_slot_list[t+1][state]

                    if adj_feature_num > 0:
                        adj_feature /= adj_feature_num
                    feature += (self.params_p[:, 1] * adj_feature.squeeze()).unsqueeze(1)
                
                # add incident edge feature
                feature = self.add_incident_edge_feature(u, feature)
                 
                features_different_slot_list[t][current_state] = feature
        assert feature.shape == (self.dim, 1)
        return feature.squeeze(1)
