import numpy as np

class makeEnvironment():

    s = np.array([[1,5,0,8,0,3,4,0,6],
                  [2,0,0,1,6,0,0,0,3],
                  [3,9,0,0,5,4,0,1,0],
                  [9,0,8,5,0,0,3,7,0],
                  [0,0,0,4,8,2,1,8,0],
                  [0,0,5,0,0,0,9,6,0],
                  [0,7,0,0,0,8,0,3,0],
                  [0,0,0,0,0,5,0,4,9],
                  [0,0,9,0,0,1,8,0,7]])

    def __init__(self):
        pass

    def step(self, *actions):
        for action in actions:
            (i,j), value = action
            self.s[i][j] = value

    def action_space(self):
        return [(x//9, x % 9, i) for x in np.where(np.hstack(self.s) == 0)[0] for i in range(1,10)]
    
    def n(self):
        return len(self.action_space())

    def sample(self):
        return self.action_space()[np.random.randint(self.n())]
    

env = makeEnvironment()
print(env.action_space())

print(env.sample())
