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
from trainer import *

G = nx.DiGraph()
G.add_edge(0, 1, cost=2.0, time=1.0)
G.add_edge(0, 2, cost=1.0, time=2.0)
G.add_edge(1, 3, cost=1.0, time=1.0)
G.add_edge(2, 3, cost=1.0, time=1.0) 
node_list = list(G.nodes) 

config = dict(
    graph=G,
    feature_dim=2,
    iteration_radius=2,
    miss_deadline_penalty=10        
)

default_config = dict(
    max_iteration=100,
    max_episode_length=10,
    evaluate_interval=3,
    learning_rate=1e-3,
    gamma=0.99,
    eps=0.3,
    seed=0
)

struct2vec_config = merge_config(dict(
    memory_size=50,
    learn_start=10,
    batch_size=3,
    feature_dim=7,
    target_update_freq=5,  # in steps
    learn_freq=1,  # in steps
    clip_norm=1e-1, 
    params_init_scale=1e-1,
    n=1,
    env_class=DelayConstrainedNetworkRoutingEnv,
    env_name="DelayConstrainedNetworkRoutingEnv",
    q_value_class=GraphFeatureQValue
), default_config)

struct2vec_config = merge_config(
    struct2vec_config,
    config
)


class TestStruct2VecTrainer(unittest.TestCase):

    def test_initialize_parameters(self):
        struct2vec_trainer = Struct2VecTrainer(struct2vec_config)
        struct2vec_trainer.initialize_parameters()

    def test_compute_q_value(self):
        struct2vec_trainer = Struct2VecTrainer(struct2vec_config)
        state = (0, 3, 2.5)
        next_state = (1, 3, 1.5) 
        q_value = struct2vec_trainer.compute_q_value(state, next_state)
        self.assertIsInstance(q_value, np.ndarray)

    def test_get_maximum_q_value(self):
        struct2vec_trainer = Struct2VecTrainer(struct2vec_config)
        state = (0, 3, 2.5)
        maximum_q_value = struct2vec_trainer.get_maximum_q_value(state)
        self.assertIsInstance(maximum_q_value, torch.Tensor)

    def test_compute_action(self):
        struct2vec_trainer = Struct2VecTrainer(struct2vec_config)
        state = (0, 3, 2.5)
        act = struct2vec_trainer.compute_action(state)
        self.assertIn(act, [(0, 1), (0, 2)])
    
    def test_compute_baseline_action(self):
        struct2vec_trainer = Struct2VecTrainer(struct2vec_config)
        state = (0, 3, 2.5)
        act = struct2vec_trainer.compute_fastest_action(state)
        self.assertEqual(act, (0, 1))
        state = (3, 0, 2.5)
        act = struct2vec_trainer.compute_fastest_action(state)
        self.assertIsNone(act)

    def test_train(self):
        struc2vec_trainer = Struct2VecTrainer(struct2vec_config)
        struc2vec_trainer.train()

if __name__ == '__main__':
    unittest.main()



