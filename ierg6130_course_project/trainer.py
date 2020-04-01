"""
We implement methods for training the RL agent for delay constrained network optimization. 
"""
import gym
import numpy as np
import torch
import torch
import torch.nn as nn
import inspect 
import time
from replay_memory import *
from delay_constrained_network_env import *
from q_value import QValue
from utils import *

default_config = dict(
    max_iteration=1000,
    max_episode_length=1000,
    evaluate_interval=100,
    gamma=0.99,
    eps=0.3,
    seed=0
)

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

        values = self.compute_values(processed_state)
        assert values.ndim == 1, values.shape

        if eps is None:
            eps = self.eps

        # Implement the epsilon-greedy policy here. We have `eps`
        #  probability to choose a uniformly random action in action_space,
        #  otherwise choose action that maximizes the values.
        # Hint: Use the function of self.env.action_space to sample random
        # action.
        action = None
        coin = np.random.rand()
        if coin < eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(values)
        return action 
        #pass
        

    def evaluate(self, num_episodes=50, *args, **kwargs):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 50 episodes."""
        policy = lambda raw_state: self.compute_action(
            self.process_state(raw_state), eps=0.0)
        result = evaluate(policy, num_episodes, seed=self.seed,
                          env_name=self.env_name, *args, **kwargs)
        return result

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
    env_name="DelayConstrainedNetworkRoutingEnv"
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
        self.learning_rate = config["learning_rate"]
        self.feature_dim = config['feature_dim']
        G = config['graph']
        env_class = config['env_class'] 
        self.env = env_class(G)
        self.G = G 
        super().__init__(config)

        self.memory = ReplayMemory(config["memory_size"])
        self.learn_start = config["learn_start"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.clip_norm = config["clip_norm"]
        self.step_since_update = 0
        self.total_step = 0

    def initialize_parameters(self):
        """Initialize the pytorch model as the Q value approximator and the target Q-value approximator."""
        # initialize the Q-value approximator using QValue class  
        self.q_value_approximator = QValue(self.feature_dim) 

        self.q_value_approximator.eval()
        self.q_value_approximator.share_memory()

        # [TODO] Initialize target network, which is identical to self.network,
        # and should have the same weights with self.network. So you should
        # put the weights of self.network into self.target_network.
        self.target_network = PytorchModel(self.obs_dim, self.act_dim) #None
        self.target_network.load_state_dict(self.network.state_dict())
        #pass

        self.target_network.eval()

        # Build Adam optimizer and MSE Loss.
        # [TODO] Uncomment next few lines
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate
        )
        self.loss = nn.MSELoss()
        #pass

    def compute_values(self, processed_state):
        """Compute the value for each potential action. Note that you
        should NOT preprocess the state here."""
        # [TODO] Convert the output of neural network to numpy array
        values = self.target_network(processed_state).detach().numpy()
        return values 
        #pass

    def train(self):
        s = self.env.reset()
        processed_s = self.process_state(s)
        act = self.compute_action(processed_s)
        stat = {"loss": []}

        for t in range(self.max_episode_length):
            next_state, reward, done, _ = self.env.step(act)
            next_processed_s = self.process_state(next_state)

            # Push the transition into memory.
            self.memory.push(
                (processed_s, act, reward, next_processed_s, done)
            )

            processed_s = next_processed_s
            act = self.compute_action(next_processed_s)
            self.step_since_update += 1
            self.total_step += 1

            if done:
                break
                
            if t % self.config["learn_freq"] != 0:
                # It's not necessary to update in each step.
                continue

            if len(self.memory) < self.learn_start:
                continue
            elif len(self.memory) == self.learn_start:
                print("Current memory contains {} transitions, "
                      "start learning!".format(self.learn_start))

            batch = self.memory.sample(self.batch_size)

            # Transform a batch of state / action / .. into a tensor.
            state_batch = to_tensor(
                np.stack([transition[0] for transition in batch])
            )
            action_batch = to_tensor(
                np.stack([transition[1] for transition in batch])
            )
            reward_batch = to_tensor(
                np.stack([transition[2] for transition in batch])
            )
            next_state_batch = torch.stack(
                [transition[3] for transition in batch]
            )
            done_batch = to_tensor(
                np.stack([transition[4] for transition in batch])
            )

            with torch.no_grad():
                # [TODO] Compute the values of Q in next state in batch.
                # Hint: 
                #  1. Q_t_plus_one is the maximum value of Q values of possible
                #     actions in next state. So the input to the network is 
                #     next_state_batch.
                #  2. Q_t_plus_one is computed using the target network.
                Qs_t_plus_one = self.target_network(next_state_batch)
                Q_t_plus_one = torch.max(Qs_t_plus_one, axis=-1).values #None
                #pass
                
                assert isinstance(Q_t_plus_one, torch.Tensor)
                assert Q_t_plus_one.dim() == 1
                
                # [TODO] Compute the target value of Q in batch.
                # Hint: The Q target is simply r_t + gamma * Q_t+1 
                #  IF the episode is not done at time t.
                #  That is, the (gamma*Q_t+1) term should be masked out
                #  if done_batch[t] is True.
                #  A smart way to do so is: using (1-done_batch) as multiplier
                Q_target = reward_batch + (1-done_batch) * self.gamma * Q_t_plus_one #None
                Q_target = Q_target.squeeze()
                #pass
                assert Q_target.shape == (self.batch_size,)
            
            # [TODO] Collect the Q values in batch.
            # Hint: Remember to call self.network.train()
            #  before you get the Q value from self.network(state_batch),
            #  otherwise the graident will not be recorded by pytorch.
            self.network.train()
            Qs_t = self.network(state_batch)  #None
            Q_t = torch.autograd.Variable(torch.ones(self.batch_size), requires_grad=False)
            for i in range(self.batch_size):
                action = int(action_batch[0, i])
                Q_t[i] = Qs_t[i, action]
            #pass
    
            assert Q_t.shape == Q_target.shape

            # Update the network
            self.optimizer.zero_grad()
            loss = self.loss(input=Q_t, target=Q_target)
            loss_value = loss.item()
            stat['loss'].append(loss_value)
            loss.backward()
            
            # [TODO] Gradient clipping. Uncomment next line
            nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_norm)
            #pass
            
            self.optimizer.step()
            self.network.eval()

        if len(self.memory) >= self.learn_start and \
                self.step_since_update > self.target_update_freq:
            print("{} steps has passed since last update. Now update the"
                  " parameter of the behavior policy. Current step: {}".format(
                self.step_since_update, self.total_step
            ))
            self.step_since_update = 0
            # [TODO] Copy the weights of self.network to self.target_network.
            self.target_network.load_state_dict(self.network.state_dict())

            #pass
            
            self.target_network.eval()
            
        return {"loss": np.mean(stat["loss"]), "episode_len": t}

    def process_state(self, state):
        return torch.from_numpy(state).type(torch.float32)