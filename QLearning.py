from collections import defaultdict
import numpy as np

class Create():

    def __init__(self, env, s, learning_rate= 0.01, discount_factor= 0.01):
        self.env = env
        self.s = s
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def step(self):
        return self.s + self.learning_rate *(self.reward() + self.discount_factor* self.policy() - self.s)

    def reward(self):
        pass

    def policy(self, num_actions, eps):
        action_probabilities = np.ones(num_actions,dtype = float) * eps / num_actions
            
        best_action = np.argmax(self.Q[self.s])
        action_probabilities[best_action] += (1.0 - eps)
        return action_probabilities
