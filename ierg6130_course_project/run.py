import inspect
import time
import networkx as nx
import numpy as np
import os
from trainer import *

trainer_cls = Struct2VecTrainer
G_toy = nx.DiGraph()
G_toy.add_edge(0, 1, cost=2.0, time=1.0)
G_toy.add_edge(0, 2, cost=1.0, time=2.0)
G_toy.add_edge(1, 3, cost=1.0, time=1.0)
G_toy.add_edge(2, 3, cost=1.0, time=1.0) 
node_list = list(G_toy.nodes) 

#HOME_PATH = '/c/Users/ThinkPad/Desktop/IERG6130/ierg6130_course_project/'\
#            + 'ierg6130_course_project/'
G_medium_size = nx.read_gml('data/G_medium_size.gml',
                            destringizer=eval
                            )


G = G_medium_size
G_name = 'G_medium_size'
config = dict(
    graph=G,
    iteration_radius=3,
    miss_deadline_penalty=1 
)

default_config = dict(
    max_iteration=5000,
    max_episode_length=50,
    evaluate_interval=1,
    learning_rate=1e-3,
    gamma=0.99,
    eps=0.3,
    params_init_scale=1e-3,
    seed=0
)

struct2vec_config = merge_config(dict(
    memory_size=500,
    learn_start=1,
    batch_size=100,
    feature_dim=5,
    target_update_freq=200,  # in steps
    learn_freq=50,  # in steps
    clip_norm=1, 
    n=1,
    env_class=DelayConstrainedNetworkRoutingEnv,
    env_name="DelayConstrainedNetworkRoutingEnv",
    q_value_class=GraphFeatureQValue
), default_config)

struct2vec_config = merge_config(
    struct2vec_config,
    config
)

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
    rewards = [] 
    for i in range(config['max_iteration'] + 1):
        stat = trainer.train(use_fastest_supervisor=True)
        stats.append(stat or {})
        if i % config['evaluate_interval'] == 0 or \
                i == config["max_iteration"]:
            reward, fastest_path_reward = trainer.evaluate(config['env_class'], config, config.get("evaluate_num_episodes", 200))
            print("({:.1f}s,+{:.1f}s)\tIteration {}, current mean episode "
                  "reward is {}, current baseline mean episode reward is {}. {}".format(
                time.time() - start, time.time() - now, i, reward, fastest_path_reward,
                {k: round(np.mean(v), 4) for k, v in
                 stat.items()} if stat else ""))
            rewards.append([i, reward, fastest_path_reward]) 
            now = time.time()
        if reward_threshold is not None and reward > reward_threshold:
            print("In {} iteration, current mean episode reward {:.3f} is "
                  "greater than reward threshold {}. Congratulation! Now we "
                  "exit the training process.".format(
                i, reward, reward_threshold))
            break
    return trainer, stats, rewards


if __name__ == '__main__':
    trainer, stats, rewards = run(trainer_cls, struct2vec_config)
    np.save(time.asctime+'_stats_rewards_'+G_name, [stats, rewards])
