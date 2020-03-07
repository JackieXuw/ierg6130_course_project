"""
delay constrained network environment has the following traits in common:

- A node where the agent is currently on  
- A destination where the agent is going to
- A float type remaining time that the agent has before the deadline 

Agents can determine which edge to go and (optional) the time to pay on one edge. Observations consist
of the current node, remaining time and destination.

Actions consist of:
    - Wich edge to move on from current node  
    - (optional) How much time to pay on the selected edge 

An episode ends when:
    - The agent arrives at the destination.
    - The agent miss the deadline.

Reward schedule:
    move on one edge e: -c_e
    miss the deadline: -C (where C is a large constant)

"""
import sys
import numpy as np
import networkx as nx 
from copy import deepcopy
from space import DelayConstrainedNetworkObservationSpace
from space import DelayConstrainedNetworkActionSpace
from gym import Env
from gym.utils import colorize, seeding
from contextlib import closing
from six import StringIO


class DelayConstrainedNetworkRoutingEnv(Env):

    metadata = {}
    
    def __init__(self, graph, miss_deadline_penalty=500):
        """
        G: a networkx direct Graph, each edge is associated with a cost and a time.
        miss_deadline_penalty: the penalty incurred if the deadline is violated. 
        """
        assert isinstance(graph, nx.Graph)
        self.graph = deepcopy(graph)
        self.miss_deadline_penalty = miss_deadline_penalty
        self.current_node = None
        self.destination = None
        self.remaining_time = None 
        self.last_action = None 
        self.last_reward = None 
        self.num_step = 0 
        # Cumulative reward earned this episode
        self.episode_total_reward = None

        # the action_space is state-dependent and thus we implement a get_action_space as a lambda function to get the list of next nodes
        self.get_action_space = lambda node: DelayConstrainedNetworkActionSpace([ (node, next_node) for next_node \
                                                in list(nx.bfs_successors(self.graph, node, depth_limit=1))[0][1]]) 

        self.observation_space = DelayConstrainedNetworkObservationSpace(self.graph)  
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_observation(self):
        """Return a string representation of the current state."""
        raise NotImplementedError

    def render(self, mode='human'):
        """
        Draw the networkx graph and show the source and destination in different colors.
        """
        raise NotImplementedError
        
    def step(self, action):
        assert self.current_node is not None
        assert self.destination is not None
        assert type(self.remaining_time) in [int, float]
        assert self.remaining_time >= 0 
        current_node = self.current_node
        destination = self.destination
        remaining_time = self.remaining_time
        action_space = self.get_action_space(current_node) 
        assert action_space.contains(action)
        _, next_node = action
         
        done = False
        reward = 0.0
        self.num_step += 1

        if current_node == destination:
            obs = (current_node, destination, remaining_time)  # if we are already at the destination, we do nothing 
            done = True
        else:
            act_time = self.graph[current_node][next_node]['time']
            act_cost = self.graph[current_node][next_node]['cost'] 
            reward += (-act_cost)  
            remaining_time -= act_time
            obs = (next_node, destination, remaining_time)
            arrive = (next_node == destination)
            miss_ddl = (remaining_time < 0)

            if miss_ddl:
                reward += (-self.miss_deadline_penalty)
            
            if miss_ddl or arrive:
                done = True
            current_node = next_node

        self.last_action = action
        self.last_reward = reward 
        self.episode_total_reward += reward
        
        # update the current state
        self.current_node = current_node
        self.remaining_time = remaining_time
        return (obs, reward, done, {})

    def reset(self, init_state=None):
        self.last_action = None
        self.last_reward = 0
        if init_state is None:
            current_node, destination, remaining_time = self.observation_space.sample()
        else:
            current_node, destination, remaining_time = init_state

        self.current_node = current_node
        self.destination = destination
        self.remaining_time = remaining_time 
        self.episode_total_reward = 0.0
        self.num_step = 0
        return self._get_obs()

    def _get_obs(self):
        return (self.current_node, self.destination, self.remaining_time)

    def _move(self, movement):
        raise NotImplementedError


