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
from utils import *


def run(trainer_cls, config=None, reward_threshold=None):
    """Run the trainer and report progress, agnostic to the class of trainer
    :param trainer_cls: A trainer class 
    :param config: A dict
    :param reward_threshold: the reward threshold to break the training
    :return: The trained trainer and a dataframe containing learning progress
    """
    assert inspect.isclass(trainer_cls)
    if config is None:
        config = {}
    trainer = trainer_cls(config)
    config = trainer.config
    start = now = time.time()
    stats = []
    for i in range(config['max_iteration'] + 1):
        stat = trainer.train()
        stats.append(stat or {})
        if i % config['evaluate_interval'] == 0 or \
                i == config["max_iteration"]:
            reward = trainer.evaluate(config.get("evaluate_num_episodes", 50))
            print("({:.1f}s,+{:.1f}s)\tIteration {}, current mean episode "
                  "reward is {}. {}".format(
                time.time() - start, time.time() - now, i, reward,
                {k: round(np.mean(v), 4) for k, v in
                 stat.items()} if stat else ""))
            now = time.time()
        if reward_threshold is not None and reward > reward_threshold:
            print("In {} iteration, current mean episode reward {:.3f} is "
                  "greater than reward threshold {}. Congratulation! Now we "
                  "exit the training process.".format(
                i, reward, reward_threshold))
            break
    return trainer, stats

default_config = dict(
    env_name="DelayConstrainedNetworkRoutingEnv",
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

    def __init__(self, config, env_class):
        self.config = merge_config(config, default_config)

        # Create the environment
        self.env_name = self.config['env_name']
        G = config['graph'] 
        self.env = env_class(G)
        
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
mlp_trainer_config = merge_config(dict(
    parameter_std=0.01,
    learning_rate=0.01,
    hidden_dim=100,
    n=3,
    clip_norm=1.0,
    clip_gradient=True
), default_config)


class MLPTrainer(LinearTrainer):
    def __init__(self, config):
        config = merge_config(config, mlp_trainer_config)
        self.hidden_dim = config["hidden_dim"]
        super().__init__(config)

    def initialize_parameters(self):
        # [TODO] Initialize self.hidden_parameters and self.output_parameters,
        #  which are two dimensional matrices, and subject to normal
        #  distributions with scale config["parameter_std"]
        std = self.config["parameter_std"]
        self.hidden_parameters = std * np.random.randn(self.obs_dim, self.hidden_dim) #None
        self.output_parameters = std * np.random.randn(self.hidden_dim, self.act_dim) #None
        #pass

    def compute_values(self, processed_state):
        """Compute the value for each potential action. Note that you
        should NOT preprocess the state here."""
        assert processed_state.ndim == 1, processed_state.shape
        activation = self.compute_activation(processed_state)
        values = np.matmul(self.output_parameters.transpose(), activation)#None
        #pass
        return values 

    def compute_activation(self, processed_state):
        """Given a processed state, first we need to compute the activtaion
        (the output of hidden layer). Then we compute the values (the output of
        the output layer).
        """
        pre_activation = np.matmul(self.hidden_parameters.transpose(), processed_state)
        activation_function = lambda x: np.tanh(x)
        activation = activation_function(pre_activation) #None
        return activation
        #pass

    def compute_gradient(self, processed_states, actions, rewards, tau, T):
        n = self.n
        
        # [TODO] compute the target value.
        # Hint: copy your codes in LinearTrainer.
        G = 0 #None
        #pass
        for i in range(min(n,T-tau)):
            try:
                G += (self.gamma**i)*rewards[tau+1+i]
            except Exception as e:
                print('Error in accessing reward! {}'.format(e))
                break 
        
        #pass

        if tau + n < T:
            # Hint: Since we use Sarsa algorithm here,
            #  the Q value of time tau+n is the Q value of action
            #  in time tau+n. So you should take the tau+n element of
            #  processed_states as input to compute the Q values
            #  and then take the "actions[tau+n]" as the index to get
            #  the Q value in tau+n.
            processed_state_step_n = processed_states[tau+n]
            action_step_n = actions[tau+n]
            Q_tau_plus_n = self.compute_values(processed_state_step_n)[action_step_n] #None
            #pass
            
            G = G + (self.gamma ** n) * Q_tau_plus_n

        # Denote the state-action value function Q, then the loss of
        # prediction error w.r.t. the output layer weights can be 
        # separated into two parts (the chain rule):
        #     dError / dweight = (dError / dQ) * (dQ / dweight)
        # We call the first one loss_grad, and the latter one
        # value_grad. We consider the Mean Square Error between the target
        # value (G) and the predict value (Q(s_t, a_t)) to be the loss.
        cur_state = processed_states[tau]
        loss_grad = np.zeros((self.act_dim, 1))  # [act_dim, 1]
        # [TODO] compute loss_grad
        cur_action =  actions[tau]
        cur_value_approximation = self.compute_values(cur_state)[cur_action]
        loss_grad[cur_action] = -(G-cur_value_approximation)
        #pass
        
        # [TODO] compute the gradient of output layer parameters
        hidden_layer_output = self.compute_activation(cur_state)
        output_gradient = np.zeros((self.hidden_dim, self.act_dim)) #None
        output_gradient[:, cur_action] = loss_grad[cur_action] * hidden_layer_output 
        
        #pass
        
        # [TODO] compute the gradient of hidden layer parameters
        # Hint: using chain rule and derive the formulation
        activation_gradient_function = lambda x : -np.tanh(x)**2 + 1
        hidden_layer_output_gradient = loss_grad[cur_action] * self.output_parameters[:, cur_action]

        pre_activation = np.matmul(self.hidden_parameters.transpose(), cur_state)
        activation_gradient = activation_gradient_function(pre_activation)
        hidden_layer_pre_activation_gradient = hidden_layer_output_gradient * activation_gradient
        
        hidden_gradient = np.matmul(np.expand_dims(cur_state,axis=1), np.expand_dims(hidden_layer_pre_activation_gradient,axis=0))  #None
        #pass
    
        assert np.all(np.isfinite(output_gradient)), \
            "Invalid value occurs in output_gradient! {}".format(
                output_gradient)
        assert np.all(np.isfinite(hidden_gradient)), \
            "Invalid value occurs in hidden_gradient! {}".format(
                hidden_gradient)
        return [hidden_gradient, output_gradient]

    def apply_gradient(self, gradients):
        """Apply the gradientss to the two layers' parameters."""
        assert len(gradients) == 2
        hidden_gradient, output_gradient = gradients

        assert output_gradient.shape == (self.hidden_dim, self.act_dim)
        assert hidden_gradient.shape == (self.obs_dim, self.hidden_dim)
        
        # [TODO] Implement the clip gradient mechansim
        # Hint: when the old gradient has norm less that clip_norm,
        #  then nothing happens. Otherwise shrink the gradient to
        #  make its norm equal to clip_norm.
        if self.config["clip_gradient"]:
            clip_norm = self.config["clip_norm"]
            flattened_hidden_gradient = hidden_gradient.flatten()
            flattened_output_gradient = output_gradient.flatten()
            hidden_gradient_norm = np.linalg.norm(hidden_gradient)
            output_gradient_norm = np.linalg.norm(output_gradient)
            if hidden_gradient_norm > clip_norm:
                hidden_gradient *= (clip_norm/hidden_gradient_norm)
            if output_gradient_norm > clip_norm:
                output_gradient *= (clip_norm/output_gradient_norm)
            #gradient_norm = np.linalg.norm(flattened_gradients) 
            #if clip_norm <  gradient_norm:
            #    output_gradient *= (clip_norm/gradient_norm)
            #    hidden_gradient *= (clip_norm/gradient_norm)
            #pass

        # [TODO] update the parameters
        # Hint: Remember to check the sign when applying the gradient
        #  into the parameters. Should you add or minus the gradients?
        self.output_parameters = self.output_parameters - self.learning_rate * output_gradient
        self.hidden_parameters = self.hidden_parameters - self.learning_rate * hidden_gradient
        
        # import ipdb; ipdb.set_trace()

        #pass
