import json  
from copy import copy

import numpy as np
import networkx as nx
import unittest 
from space import DelayConstrainedNetworkObservationSpace
from space import DelayConstrainedNetworkActionSpace

class TestDelayConstrainedNetworkObservationSpace(unittest.TestCase):

    def test_sample(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, cost=2.0, time=1.0)
        G.add_edge(0, 2, cost=1.0, time=2.0)
        G.add_edge(1, 3, cost=1.0, time=1.0)
        G.add_edge(2, 3, cost=1.0, time=1.0) 
        node_list = list(G.nodes) 
        delay_constrained_net_obs_space = DelayConstrainedNetworkObservationSpace(G)
        delay_ratio_lw = 1.0
        delay_ratio_up = 2.0
        current_node, destination, remaining_time = delay_constrained_net_obs_space.sample()
        self.assertIn(current_node, node_list)
        self.assertIn(destination, node_list)
        reachable = nx.has_path(G, current_node, destination)
        if reachable:
            fastest_path_time = nx.shortest_path_length(G, source=current_node, target=destination)
            delay_lw = fastest_path_time * delay_ratio_lw
            delay_up = fastest_path_time * delay_ratio_up 
            self.assertGreater(remaining_time, delay_lw)
            self.assertLess(remaining_time, delay_up) 
        else:
            self.assertEqual(remaining_time, 0)
    
    def test_contains(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, cost=2.0, time=1.0)
        G.add_edge(0, 2, cost=1.0, time=2.0)
        G.add_edge(1, 3, cost=1.0, time=1.0)
        G.add_edge(2, 3, cost=1.0, time=1.0)
        node_list = list(G.nodes)
        delay_constrained_net_obs_space = DelayConstrainedNetworkObservationSpace(G)
        for node_1 in node_list:
            for node_2 in node_list:
                self.assertTrue(delay_constrained_net_obs_space.contains((node_1, node_2, 1)))
        self.assertFalse(delay_constrained_net_obs_space.contains((1, 3, -1)))
            

class TestDelayConstrainedNetworkActionSpace(unittest.TestCase):

    def test_sample(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, cost=2.0, time=1.0)
        G.add_edge(0, 2, cost=1.0, time=2.0)
        G.add_edge(1, 3, cost=1.0, time=1.0)
        G.add_edge(2, 3, cost=1.0, time=1.0) 
        node_list = list(G.nodes)
        for node in node_list:
            edge_list = list(G.out_edges(node))
            action_space = DelayConstrainedNetworkActionSpace(edge_list)
            act = action_space.sample() 
            if act is None and len(edge_list) == 0:
                continue
            self.assertIn(act, edge_list)
        
    def test_contains(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, cost=2.0, time=1.0)
        G.add_edge(0, 2, cost=1.0, time=2.0)
        G.add_edge(1, 3, cost=1.0, time=1.0)
        G.add_edge(2, 3, cost=1.0, time=1.0)
        node_list = list(G.nodes)
        for node in node_list:
            edge_list = list(G.out_edges(node))
            action_space = DelayConstrainedNetworkActionSpace(edge_list)
            for edge in edge_list:
                self.assertTrue(action_space.contains(edge))
            self.assertFalse(action_space.contains((3,4))) 
            

if __name__ == '__main__':
    unittest.main()





