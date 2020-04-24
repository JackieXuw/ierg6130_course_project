import json  
from copy import copy

import numpy as np
import networkx as nx
import unittest 
from delay_constrained_network_env import DelayConstrainedNetworkRoutingEnv

G = nx.DiGraph()
G.add_edge(0, 1, cost=2.0, time=1.0)
G.add_edge(0, 2, cost=1.0, time=2.0)
G.add_edge(1, 3, cost=1.0, time=1.0)
G.add_edge(2, 3, cost=1.0, time=1.0) 
node_list = list(G.nodes) 

class TestDelayConstrainedNetworkRoutingEnv(unittest.TestCase):

    def test_reset(self):
        delay_constrained_net_env = DelayConstrainedNetworkRoutingEnv(G)
        delay_ratio_lw = 1.0
        delay_ratio_up = 2.0
        reset_time = len(node_list)
        for i in range(reset_time):
            current_node, destination, remaining_time = delay_constrained_net_env.reset()
            self.assertIn(current_node, node_list)
            self.assertIn(destination, node_list)
            reachable = nx.has_path(G, current_node, destination)
            if reachable:
                fastest_path_time = nx.shortest_path_length(G, source=current_node, target=destination, weight='time')
                delay_lw = fastest_path_time * delay_ratio_lw
                delay_up = fastest_path_time * delay_ratio_up 
                self.assertGreater(remaining_time, delay_lw)
                self.assertLess(remaining_time, delay_up) 
            else:
                self.assertEqual(remaining_time, 0)

    def test_step(self):
        miss_ddl_penalty = 1
        delay_constrained_net_env = DelayConstrainedNetworkRoutingEnv(G, miss_ddl_penalty)
        delay_constrained_net_env.reset(init_state=(0, 3, 2.5)) 
        
        act_1 = (0,2)
        obs, reward, done, _ = delay_constrained_net_env.step(act_1)
        self.assertEqual(obs, (2, 3, 0.5))
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)
        
        act_2 = (2,3)
        obs, reward, done, _ = delay_constrained_net_env.step(act_2)
        self.assertEqual(obs, (3, 3, -0.5))
        self.assertEqual(reward, - miss_ddl_penalty * 1.0 - 1.0 - 1.0)
        self.assertTrue(done)

        final_obs = delay_constrained_net_env._get_obs()
        self.assertEqual(final_obs, (3, 3, -0.5))
    
    def test_find_next_action_states(self):
        delay_constrained_net_env = DelayConstrainedNetworkRoutingEnv(G)
        init_state = (0, 3, 2.5)
        action_list, next_states_list = delay_constrained_net_env.\
            find_next_action_states(init_state)
        self.assertEqual(set(action_list), {(0, 1), (0, 2)})
        self.assertEqual(set(next_states_list), {(1, 3, 1.5), (2, 3, 0.5)})

    def test_set_cost_time_radius(self):
        delay_constrained_net_env = DelayConstrainedNetworkRoutingEnv(G)
        time_radius = delay_constrained_net_env.time_radius
        cost_radius = delay_constrained_net_env.cost_radius
        self.assertEqual(time_radius, 2.0)
        self.assertEqual(cost_radius, 2.0)



if __name__ == '__main__':
    unittest.main()




