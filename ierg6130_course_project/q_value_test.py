import json  
from copy import copy
import torch
import numpy as np
from numpy.testing import assert_almost_equal
from utils import *
import networkx as nx
import unittest 
from q_value import GraphFeatureQValue 
from delay_constrained_network_env import DelayConstrainedNetworkRoutingEnv 

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


class TestGraphFeatureQValue(unittest.TestCase):

    def test_forward(self):
        feature_dim = config['feature_dim']
        G = config['graph'] 
        graph_feature_q_value = GraphFeatureQValue(G, config)
        delay_constrained_network_routing_env = \
            DelayConstrainedNetworkRoutingEnv(G) 
        test_state_num = 10
        for i in range(test_state_num):
            state = delay_constrained_network_routing_env.reset()
            curr_node = state[0]
            action_space = \
                delay_constrained_network_routing_env.get_action_space(state[0])
            act = action_space.sample()
            if act is None:
                continue 
            obs, reward, done, _ = delay_constrained_network_routing_env.step(act)
            q_value = graph_feature_q_value(state, obs)
         
            self.assertIsInstance(q_value, torch.Tensor) 
         
if __name__ == '__main__':
    unittest.main()




