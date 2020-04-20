import json  
from copy import copy
import torch
import numpy as np
from numpy.testing import assert_almost_equal
import networkx as nx
import unittest 
from feature import GraphFeature

G = nx.DiGraph()
G.add_edge(0, 1, cost=2.0, time=1.0)
G.add_edge(0, 2, cost=1.0, time=2.0)
G.add_edge(1, 3, cost=1.0, time=1.0)
G.add_edge(2, 3, cost=1.0, time=1.0) 
node_list = list(G.nodes) 

config = {
    'graph': G,
    'feature_dim':2,
    'iteration_radius':2,
    'miss_deadline_penalty':10        
}

class TestGraphFeature(unittest.TestCase):

    def test_get_fastest_time(self):
        graph_feature = GraphFeature(config)
        self.assertAlmostEqual(graph_feature.get_fastest_time(0, 3), 2.0)
        self.assertAlmostEqual(graph_feature.get_fastest_time(3, 1), config['miss_deadline_penalty'])
            
    def test_forward(self):
        graph_feature = GraphFeature(config)
        state_1 = (0, 3, 3.0)
        feature_1 = graph_feature(state_1)
        self.assertIsInstance(feature_1, torch.Tensor) 
        
if __name__ == '__main__':
    unittest.main()



