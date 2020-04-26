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

    def __init__(self, graph, miss_deadline_penalty=1):
        """
        G: a networkx direct Graph, each edge is associated with a cost and a time.
        miss_deadline_penalty: the penalty incurred if the deadline is violated. 
        """
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.edges()) > 0
        self.graph = deepcopy(graph)
        self.set_cost_time_radius() 
        self.miss_deadline_penalty = miss_deadline_penalty
        self.current_node = None
        self.destination = None
        self.remaining_time = None 
        self.last_action = None 
        self.last_reward = None 
        self.num_step = 0 
        # Cumulative reward earned this episode
        self.episode_total_reward = None

        # the action_space is state-dependent and thus we implement a 
        # get_action_space as a lambda function to get the list of next nodes
        self.get_action_space = \
            lambda node: DelayConstrainedNetworkActionSpace([(node, next_node) for next_node \
                         in list(nx.bfs_successors(self.graph, node, depth_limit=1))[0][1]]) 

        self.observation_space = DelayConstrainedNetworkObservationSpace(self.graph)  
        self.seed()
        self.reset()
    
    def set_cost_time_radius(self):
        graph = self.graph
        node2node_cost = dict()
        node2node_time = dict()
        node2node_fast_path_cost = dict() 
        max_cost_path_cost = 0 
        max_time_path_time = 0 
        node2node_min_costs = nx.shortest_path_length(graph, weight='cost')
        node2node_min_times = nx.shortest_path_length(graph, weight='time')
        node2node_min_time_paths = nx.shortest_path(graph, weight='time')
        for u2node_min_costs in node2node_min_costs:
            u = u2node_min_costs[0]
            for v in u2node_min_costs[1].keys():
                node2node_cost[(u,v)] = u2node_min_costs[1][v]
                max_cost_path_cost = max(max_cost_path_cost, u2node_min_costs[1][v])
        
        for u2node_min_times in node2node_min_times:
            u = u2node_min_times[0]
            for v in u2node_min_times[1].keys():
                node2node_time[(u,v)] = u2node_min_times[1][v]
                max_time_path_time = max(max_time_path_time, u2node_min_times[1][v])
                path_len = len(node2node_min_time_paths[u][v])
                cost_sum = 0
                for i in range(path_len-1):
                    s = node2node_min_time_paths[u][v][i]
                    d = node2node_min_time_paths[u][v][i+1]
                    cost_sum += graph[s][d]['cost']
                node2node_fast_path_cost[(u,v)] = cost_sum
        assert max_cost_path_cost > 0
        assert max_time_path_time > 0
        self.cost_radius = max_cost_path_cost
        self.time_radius = max_time_path_time
        self.node2node_cost = node2node_cost
        self.node2node_time = node2node_time
        self.node2node_fast_path_cost = node2node_fast_path_cost
        return max_cost_path_cost, max_time_path_time

    def find_next_action_states(self, state):
        current_node, destination, remaining_time = state
        
        if current_node is None:
            return None, None
        actions_list = [(current_node, next_node) for next_node
                        in list(nx.bfs_successors(self.graph, current_node,
                                depth_limit=1))[0][1]]
        next_states_list = []
        for action in actions_list:
            _, next_node = action
            act_time = self.graph[current_node][next_node]['time']
            next_state = (next_node, destination, remaining_time-act_time)
            next_states_list.append(next_state)

        return actions_list, next_states_list

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
                time_penalty = self.time_radius
                cost_penalty = self.cost_radius
                if nx.has_path(self.graph, current_node, destination):
                    time_penalty = nx.shortest_path_length(self.graph,
                                                           source=current_node,
                                                           target=destination,
                                                           weight='time'
                                                           )
                    cost_penalty = nx.shortest_path_length(self.graph,
                                                           source=current_node,
                                                           target=destination,
                                                           weight='cost'
                                                           )
                reward += (- time_penalty * self.miss_deadline_penalty
                           - cost_penalty
                           )
            
            if miss_ddl or arrive:
                done = True
            current_node = next_node

        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        
        # update the current state
        self.current_node = current_node
        self.remaining_time = remaining_time

        # if we are stuck in some node, then we are done 
        next_action_space = self.get_action_space(current_node) 
        if next_action_space.sample() is None:
            if done is False: 
                done = True
                # if we get stuck, then are doomed to be late
                time_penalty = self.time_radius
                cost_penalty = self.cost_radius
                reward += (- time_penalty * self.miss_deadline_penalty
                           - cost_penalty
                           )

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


