"""
The observation space of delay constrained network optimization problem include:
1. Current node 
2. Remaining time
3. Destination
"""
from gym import Space 
import networkx as nx 
import random
import numpy as np

class DelayConstrainedNetworkObservationSpace(Space):
    r"""The observation space of delay constrained network optimization problem.
    """
    def __init__(self, graph):
        assert isinstance(graph, nx.Graph) 
        self.graph = graph
        super(Space, self).__init__()

    def sample(self, delay_ratio_lw=1.0, delay_ratio_up=2.0, time_weight='time'):
        """
        Sample current node and destination in the whole graph uniformly randomly.
        After sampling current node and destination,  sample the remaining time in the range 
            [delay_ratio_lw * fastest_path_time, delay_ratio_up * fastest_path_time].
        """
        G = self.graph
        current_node, destination = random.sample(G.nodes, 2)
        reachable = nx.has_path(G, current_node, destination)    
        
        if not reachable:
            # if destination is not reachable, then set remaining_time to 0 
            return (current_node, destination, 0) 
        
        fastest_path_time = nx.shortest_path_length(G=G, source=current_node, target=destination, weight=time_weight)
        coin = np.random.rand() 
        delay_ratio = delay_ratio_lw + coin * (delay_ratio_up - delay_ratio_lw)
        remaining_time = delay_ratio * fastest_path_time 
        return (current_node, destination, remaining_time) 

    def contains(self, x):
        
        if len(x) != 3:
            return False
        
        current_node, destination, remaining_time = x

        if not type(remaining_time) in [int, float]:
             return False
        return current_node in self.graph.nodes and destination in self.graph.nodes and remaining_time >= 0 

    def __repr__(self):
        return "DelayConstrainedNetworkObservationSpace({})".format(self.graph.__repr__()) 


class DelayConstrainedNetworkActionSpace(Space):
    r"""The action space of delay constrained network optimization problem.
    """
    def __init__(self, edge_list):
        assert isinstance(edge_list, list)
        self.action_list = edge_list 
        super(Space, self).__init__()

    def sample(self):
        """
        Sample one edge from the edge_list. 
        """
        if len(self.action_list) == 0:
            return None
        act = random.sample(self.action_list, 1)[0] 
        return act 

    def contains(self, x):
        return x in self.action_list 

    def __repr__(self):
        return "DelayConstrainedNetworkActionSpace({})".format(self.action_list.__repr__()) 

 