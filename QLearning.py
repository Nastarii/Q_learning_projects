import numpy as np

class Create():

    def __init__(self, Q, learning_rate= 0.001):
        self.Q = Q
        self.learning_rate = learning_rate
    
    def step(self):
        return self.Q + self.learning_rate *(self.reward() + self.discount_factor* self.policy() - self.Q)

    def reward(self):
        pass

    def policy(self):
        pass

