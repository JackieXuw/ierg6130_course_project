"""
We implement methods for training the RL agent for delay constrained network optimization. 
"""
import gym
import numpy as np
import torch
import torch
import torch.nn as nn
import inspect 
import random
import time
import networkx as nx
from copy import deepcopy
from replay_memory import *
from delay_constrained_network_env import *
from q_value import *
from utils import *

INFINITY = 1e10
default_config = dict(
    max_iteration=1000,
    max_episode_length=1000,
    evaluate_interval=100,
    gamma=0.99,
    eps=0.3,
    seed=0
)

def evaluate(policy, env_cls, env_config, num_episodes=1, seed=0,
             render=False):
    """This function evaluate the given policy and return the mean episode 
    reward.
    :param policy: a function whose input is the observation
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :param env_cls: the class of the environment
    :env_config: the configured parameters of the environment
    :param render: a boolean flag indicating whether to render policy
    :return: the averaged episode reward of the given policy.
    """
    graph = env_config['graph']
    miss_deadline_penalty = env_config['miss_deadline_penalty']
    env = env_cls(graph, miss_deadline_penalty)
    env.seed(seed)
    rewards = []
    if render: num_episodes = 1
    for i in range(num_episodes):
        obs = env.reset()
        act = policy(obs)
        ep_reward = 0
        while True:
            if act is None:
                break
            obs, reward, done, info = env.step(act)
            act = policy(obs)
            ep_reward += reward
            if render:
                env.render()
                wait(sleep=0.05)
            if done:
                break
        rewards.append(ep_reward)
    if render:
        env.close()
    return np.mean(rewards)

class AbstractTrainer:
    """This is the abstract class for value-based RL trainer. We will inherent
    the specify algorithm's trainer from this abstract class, so that we can
    reuse the codes.
    """

    def __init__(self, config):
        self.config = merge_config(config, default_config)
        env_class = config['env_class'] 
        # Create the environment
        self.env_name = self.config['env_name']
        
        # Apply the random seed
        self.seed = self.config["seed"]
        np.random.seed(self.seed)
        #self.env.seed(self.seed)

        # We set self.obs_dim to the number of possible observation
        # if observation space is discrete, otherwise the number
        # of observation's dimensions. The same to self.act_dim.
        self.eps = self.config['eps']

        # You need to setup the parameter for your function approximator.
        self.initialize_parameters()

    def initialize_parameters(self):
        self.parameters = None
        raise NotImplementedError(
            "You need to override the "
            "Trainer._initialize_parameters() function.")

    def process_state(self, state):
        """Preprocess the state (observation) if necessary"""
        processed_state = state
        return processed_state

    def compute_values(self, processed_state):
        """Approximate the state value of given state.
        This is a private function.
        Note that you should NOT preprocess the state here.
        """
        raise NotImplementedError("You need to override the "
                                  "Trainer.compute_values() function.")

    def compute_action(self, processed_state, eps=None):
        """Compute the action given the state. Note that the input
        is the processed state."""

        raise NotImplementedError("You need to override the "
                                  "Trainer.compute_action() function.")        

    def compute_baseline_action(self, processed_state, eps=None):
        """Compute the baseline action given the state. Note that the input
        is the processed state."""

        raise NotImplementedError("You need to override the "
                                  "Trainer.compute_action() function.")        


    def evaluate(self, env_cls, env_config, num_episodes=50, *args, **kwargs):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 50 episodes."""
        policy = lambda raw_state: self.compute_action(
            self.process_state(raw_state), eps=0.0)
        result = evaluate(policy, env_cls, env_config, num_episodes, 
                          seed=self.seed, *args, **kwargs)

        baseline_policy = lambda raw_state: self.compute_random_feasible_action(
                        self.process_state(raw_state), eps=1.0)
        baseline_result = evaluate(baseline_policy, env_cls, env_config, 
                                   num_episodes, seed=self.seed,
                                   *args, **kwargs)


        return result, baseline_result

    def compute_gradient(self, *args, **kwargs):
        """Compute the gradient."""
        raise NotImplementedError(
            "You need to override the Trainer.compute_gradient() function.")

    def apply_gradient(self, *args, **kwargs):
        """Compute the gradient"""
        raise NotImplementedError(
            "You need to override the Trainer.apply_gradient() function.")

    def train(self):
        """Conduct one iteration of learning."""
        raise NotImplementedError("You need to override the "
                                  "Trainer.train() function.")

# Build the algorithm-specify config.
struct2vec_config = merge_config(dict(
    memory_size=50000,
    learn_start=5000,
    batch_size=32,
    feature_dim=7,
    target_update_freq=500,  # in steps
    learn_freq=1,  # in steps
    n=1,
    env_class=DelayConstrainedNetworkRoutingEnv,
    env_name="DelayConstrainedNetworkRoutingEnv",
    q_value_class=GraphFeatureQValue
), default_config)


def to_tensor(x):
    """A helper function to transform a numpy array to a Pytorch Tensor"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).type(torch.float32)
    assert isinstance(x, torch.Tensor)
    if x.dim() == 3 or x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2 or x.dim() == 4, x.shape
    return x


class Struct2VecTrainer(AbstractTrainer):
    def __init__(self, config):
        config = merge_config(config, struct2vec_config)
        assert 'graph' in config.keys()
        self.config = config
        self.learning_rate = config["learning_rate"]
        self.feature_dim = config['feature_dim']
        self.gamma = config['gamma'] 
        G = config['graph']
        env_class = config['env_class'] 
        self.env = env_class(G, miss_deadline_penalty=config['miss_deadline_penalty'])
        self.config['time_radius'] = self.env.time_radius
        self.config['cost_radius'] = self.env.cost_radius 
        self.config['node2node_cost'] = self.env.node2node_cost
        self.config['node2node_time'] = self.env.node2node_time 
        self.config['node2node_fast_path_cost'] = self.env.node2node_fast_path_cost  
        self.G = G 
        self.QValue = config['q_value_class']
        super().__init__(config)

        self.memory = ReplayMemory(config["memory_size"])
        self.learn_start = config["learn_start"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.clip_norm = config["clip_norm"]
        self.step_since_update = 0
        self.total_step = 0
        self.max_episode_length = config['max_episode_length']

    def initialize_parameters(self):
        """Initialize the pytorch model as the Q value approximator and the target Q-value approximator."""
        # initialize the Q-value approximator using QValue
        # class
        QValue = self.QValue
        self.q_value_approximator = QValue(self.G, self.config) 
        self.q_value_approximator.eval()
        self.q_value_approximator.share_memory()

        # initialize target approximator, which is identical
        # to self.q_value_approximator,
        # and should have the same weights with self.q_value_approximator.
        # So you should
        # put the weights of self.q_value_approximator into 
        # self.target_q_value_approximator.
        self.target_q_value_approximator = QValue(self.G, self.config)
        self.target_q_value_approximator.load_state_dict(self.q_value_approximator.state_dict())

        self.target_q_value_approximator.eval()

        # Build Adam optimizer and MSE Loss.
        self.optimizer = torch.optim.Adam(
            self.q_value_approximator.parameters(), lr=self.learning_rate
        )
        self.feature_optimizer = torch.optim.Adam(
            self.q_value_approximator.graph_feature.parameters(), lr=self.learning_rate
        )
        self.loss = nn.MSELoss()

    def compute_q_value(self, processed_state, processed_action):
        """Compute the q value for each state action pair. Note that you
        should NOT preprocess the state and action here."""
        values = self.q_value_approximator(
            processed_state, processed_action).detach().numpy()

        return values

    def get_maximum_q_value(self, processed_state):
        """
        Get the maximum q_value on current process_state.
        """
        action_list, next_state_list = self.env.find_next_action_states(
            processed_state
        )
        assert action_list is not None 
        if len(action_list) == 0:
            return self.compute_q_value(processed_state, processed_state)
        
        q_values = torch.Tensor(
            [self.compute_q_value(processed_state, next_state)
                for next_state in next_state_list])

        max_q_value = torch.max(q_values)
        
        return max_q_value
    
    def compute_action(self, processed_state, eps=None):
        action_list, next_state_list = self.env.find_next_action_states(
            processed_state
        )
        assert action_list is not None
        feasible_action_list = []
        feasible_next_state_list =[]
        for i in range(len(action_list)):
            next_node, destination, next_node_remaining_time = next_state_list[i]
            if (next_node, destination) not in self.config['node2node_time'].keys():
                continue
            if self.config['node2node_time'][(next_node, destination)] > next_node_remaining_time:
                continue
            feasible_action_list.append(action_list[i])
            feasible_next_state_list.append(next_state_list[i])
        #import pdb; pdb.set_trace() 
        action_list = deepcopy(feasible_action_list)
        next_state_list = deepcopy(feasible_next_state_list)
        if len(action_list) == 0:
            return None
        #import pdb; pdb.set_trace()
        q_value_list = [self.compute_q_value(processed_state, next_state)
                        for next_state in next_state_list]

        if eps is None:
            eps = self.eps

        # Implement the epsilon-greedy policy here. We have `eps`
        #  probability to choose a uniformly random action in action_space,
        #  otherwise choose action that maximizes the values.
        action = None
        coin = np.random.rand()
        if coin < eps:
            action = random.sample(action_list, 1)[0] 
        else:
            action_id = np.argmax(q_value_list)
            action = action_list[action_id]
        return action
    
    def compute_random_feasible_action(self, processed_state, eps=None):
        action_list, next_state_list = self.env.find_next_action_states(
            processed_state
        )
        assert action_list is not None
        feasible_action_list = []
        feasible_next_state_list =[]
        for i in range(len(action_list)):
            next_node, destination, next_node_remaining_time = next_state_list[i]
            if (next_node, destination) not in self.config['node2node_time'].keys():
                continue
            if self.config['node2node_time'][(next_node, destination)] > next_node_remaining_time:
                continue
            feasible_action_list.append(action_list[i])
            feasible_next_state_list.append(next_state_list[i])
        #import pdb; pdb.set_trace() 
        action_list = deepcopy(feasible_action_list)
        next_state_list = deepcopy(feasible_next_state_list)
        if len(action_list) == 0:
            return None
    
        action = random.sample(action_list, 1)[0] 
        return action

    def compute_baseline_action(self, processed_state, eps=None):
        current_node, destination, remaining_time = processed_state
        has_path = nx.has_path(self.G, current_node, destination)
        if not has_path:
            return None
        fastest_path = nx.shortest_path(self.G, source=current_node,
                                        target=destination)
        if len(fastest_path) == 1:
            return None 
        action = (fastest_path[0], fastest_path[1])
        return action

    def train(self, use_fastest_supervisor=False):
        s = self.env.reset()
        processed_s = self.process_state(s)
        act = self.compute_random_feasible_action(processed_s)
        stat = {"loss": []}

        for t in range(self.max_episode_length):
            if act is None:
                break 
            next_state, reward, done, _ = self.env.step(act)
            next_processed_s = self.process_state(next_state)

            # Push the transition into memory.
            if act is not None: 
                self.memory.push(
                (processed_s, act, reward, next_processed_s, done)
            )

            processed_s = next_processed_s
            act = self.compute_action(next_processed_s)
            self.step_since_update += 1
            self.total_step += 1

            # check if the environment is done 
            if done:
                break

            #import pdb; pdb.set_trace()
            if t % self.config["learn_freq"] != 0 and act is not None:
                # It's not necessary to update in each step.
                continue

            if len(self.memory) < self.learn_start:
                continue
            elif len(self.memory) == self.learn_start:
                print("Current memory contains {} transitions, "
                      "start learning!".format(self.learn_start))

            batch = self.memory.sample(self.batch_size)

            # take out the state batch, action_batch, reward_batch,
            # next_state_batch and done_batch
            state_batch = [transition[0] for transition in batch]
            action_batch = [transition[1] for transition in batch]
            reward_batch = torch.Tensor([transition[2] for transition in batch]
                                        )
            next_state_batch = [transition[3] for transition in batch]
            done_batch = torch.Tensor([transition[4] for transition in batch])

            with torch.no_grad():
                # Compute the values of Q in next state in batch.
                #  1. Q_t_plus_one is the maximum value of Q values of possible
                #     actions in next state. So the input to the network is 
                #     next_state_batch.
                #  2. Q_t_plus_one is computed using the target network.
                Q_t_plus_one = torch.Tensor([
                    self.get_maximum_q_value(next_state).item() for
                    next_state in next_state_batch])
                Q_t_plus_one = Q_t_plus_one.squeeze()
                assert isinstance(Q_t_plus_one, torch.Tensor)
                assert Q_t_plus_one.dim() == 1
                
                # Compute the target value of Q in batch.
                # The Q target is simply r_t + gamma * Q_t+1 
                #  IF the episode is not done at time t.
                #  That is, the (gamma*Q_t+1) term should be masked out
                #  if done_batch[t] is True.
                #  A smart way to do so is: using (1-done_batch) as multiplier
                Q_target = reward_batch + (1-done_batch) * self.gamma * \
                    Q_t_plus_one
                Q_target = Q_target.squeeze()
                
                assert Q_target.shape == (self.batch_size,)

            # if we use fastest path as supervisor  
            if use_fastest_supervisor:
                for i in range(len(Q_target)):
                    state = state_batch[i]
                    current_node, destination, _ = state
                    next_state = next_state_batch[i]
                    next_node, _, _ = next_state 
                    if nx.has_path(self.G, next_node, destination):
                        Q_target[i] = - self.config['node2node_fast_path_cost'][(next_node, destination)]\
                                      - self.G[current_node][next_node]['cost'] 
                
                #Q_target = (Q_target - Q_target.mean())
             
            # Collect the Q values in batch.
            #  before you get the Q value from self.network(state_batch),
            #  otherwise the graident will not be recorded by pytorch.
            self.q_value_approximator.train()
            Q_t = torch.autograd.Variable(torch.ones(self.batch_size),
                                          requires_grad=False)
            for i in range(self.batch_size):
                state = state_batch[i]
                next_state = next_state_batch[i]
                Q_t[i] = self.q_value_approximator(state, next_state) 
    
            assert Q_t.shape == Q_target.shape

            # Update the q_value approximator
            self.optimizer.zero_grad()
            #self.feature_optimizer.zero_grad()
            loss = self.loss(input=Q_t, target=Q_target)
            loss_value = loss.item()
            stat['loss'].append(loss_value)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.q_value_approximator.parameters(), self.clip_norm
                )
            # nn.utils.clip_grad_norm_(
            #     self.q_value_approximator.graph_feature.parameters(), self.clip_norm
            #     )
            
            self.feature_optimizer.step()
            self.optimizer.step()
            self.q_value_approximator.eval()
            self.q_value_approximator.graph_feature.eval() 
                    

        if len(self.memory) >= self.learn_start and \
                self.step_since_update > self.target_update_freq:
            print("{} steps has passed since last update. Now update the"
                  " parameter of the behavior policy. Current step: {}".format(
                self.step_since_update, self.total_step
            ))
            self.step_since_update = 0
            # Copy the weights of self.q_value_approximator
            # to self.target_q_value_approximator.
            self.target_q_value_approximator.load_state_dict(
                self.q_value_approximator.state_dict()
                )

            self.target_q_value_approximator.eval()

           
        return {"loss": np.mean(stat["loss"]) if stat["loss"] else -INFINITY, "episode_len": t}

    def process_state(self, state):
        return state
