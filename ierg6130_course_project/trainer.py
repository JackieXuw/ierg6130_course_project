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

def evaluate(policy, env, num_episodes=1, seed=0, render=False):
    """This function evaluate the given policy and return the mean episode 
    reward.
    :param policy: a function whose input is the observation
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :param render: a boolean flag indicating whether to render policy
    :return: the averaged episode reward of the given policy.
    """
    rewards = []
    if render: num_episodes = 1
    for i in range(num_episodes):
        obs = env.reset()
        act = policy(obs)
        ep_reward = 0
        while True:
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
    env_name="Delay",
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

        # Create the environment
        self.env_name = self.config['env_name']
        self.env = gym.make(self.env_name)
        if self.env_name == "Pong-ram-v0":
            self.env = wrap_deepmind_ram(self.env)

        # Apply the random seed
        self.seed = self.config["seed"]
        np.random.seed(self.seed)
        self.env.seed(self.seed)

        # We set self.obs_dim to the number of possible observation
        # if observation space is discrete, otherwise the number
        # of observation's dimensions. The same to self.act_dim.
        if isinstance(self.env.observation_space, gym.spaces.box.Box):
            assert len(self.env.observation_space.shape) == 1
            self.obs_dim = self.env.observation_space.shape[0]
            self.discrete_obs = False
        elif isinstance(self.env.observation_space,
                        gym.spaces.discrete.Discrete):
            self.obs_dim = self.env.observation_space.n
            self.discrete_obs = True
        else:
            raise ValueError("Wrong observation space!")

        if isinstance(self.env.action_space, gym.spaces.box.Box):
            assert len(self.env.action_space.shape) == 1
            self.act_dim = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.act_dim = self.env.action_space.n
        else:
            raise ValueError("Wrong action space!")

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

        # [TODO] Implement the epsilon-greedy policy here. We have `eps`
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