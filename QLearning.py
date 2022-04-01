from Sudoku import make
import numpy as np

class Config:

    def __init__(self, n_episodes, max_iter, ep, edd, mep, gamma, lr):
        self.n_episodes = n_episodes
        self.max_iter = max_iter
        self.exploration_proba = ep
        self.exploration_decreasing_decay = edd
        self.min_exploration_proba = mep
        self.gamma = gamma
        self.lr = lr
        self.total_reward = list()

env = make('Sudoku-v0')
n_obs, n_act = env.observation_space.n, env.action_space.n
Q_table = np.zeros((n_obs, n_act))
prev_episode = 0

config = Config(1000,1000, 1,0.001,0.01,0.99,0.1)

for episode in range(config.n_episodes):
    
    observation = env.reset()
    total_episode_reward = 0

    for i in range(config.max_iter):

        if np.random.uniform(0,1) < config.exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[observation,:])
        
        new_observation, reward, done, _ = env.step(action)
        
        Q_table[observation, action] = (1- config.lr) * Q_table[observation, action] \
        + config.lr*(reward + config.gamma*max(Q_table[new_observation,:]))
        total_episode_reward += reward
        
        if done:
            break

        observation = new_observation
    
    exploration_proba = max(config.min_exploration_proba, np.exp(-config.exploration_decreasing_decay*episode))
    config.total_reward.append(total_episode_reward)

    if episode % 200 == 199:
        print(f'Episodes Trained: [{episode}/{config.n_episodes}]\nMean Episode Reward: {np.mean(config.total_reward[prev_episode:episode]):.4f}')
        prev_episode = episode