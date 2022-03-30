from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class make:

    rst = np.array([[1,5,0,8,0,3,4,0,6],
                  [2,0,0,1,6,0,0,0,3],
                  [3,9,0,0,5,4,0,1,0],
                  [9,0,8,5,0,0,3,7,0],
                  [0,0,0,4,0,2,1,8,0],
                  [0,0,5,0,0,0,9,6,0],
                  [0,7,0,0,0,8,0,3,0],
                  [0,0,0,0,0,5,0,4,9],
                  [0,0,9,0,0,1,8,0,7]])

    GREEN_COLOR = '#487018'
    RED_COLOR = '#bc0000'
    
    def __init__(self,name):
        self.name = name
        if self.name != 'Sudoku-v0':
            raise Exception('The environment need to be Sudoku-v0')
        self.actions = []
        self.s = self.rst.copy()
        self.action_space = Discrete(self.s)
        self.observation_space = Box(self.s, self.actions)
    
    def done(self,s):
        if np.count_nonzero(s==0) == 0:
            return True
        else:
            return False
        '''
        for idx in range(len(self.s)):
            if len(np.unique(self.s[idx])) != 9:
                return False
            if len(np.unique(self.s[:,idx])) != 9:
                return False
        return True
        '''

    def info(self):
        return {'possibilities': len(self.s)**2*9,}

    def render(self,s, show=False):
        fig, ax = plt.subplots()

        fig.patch.set_visible(False)
        fig.canvas.manager.set_window_title(self.name)
        
        ax.axis('off')
        ax.axis('tight')

        df = pd.DataFrame(s)
        df[df == 0] = ''
        table = ax.table(cellText=df.values, cellLoc='center', 
                 loc='center', bbox=[0,0,1,1])
        
        for act in self.actions:
            table.get_celld()[(act[0],act[1])]._text.set_color(act[3])

        l = 0.1096
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((l/3*i -0.0548, l/3*j -0.0548),l/3,l/3, facecolor='none', edgecolor='black', linewidth=2))
        
        fig.tight_layout()

        if show:
            plt.show()
    
    def reset(self):
        self.s, self.actions = self.rst, []
        return self.observation_space

    def reward(self, x, y, val, s):
        self.color, r = self.GREEN_COLOR, 1

        if len(np.unique(s[x])) == len(s[x]) or \
           len(np.unique(s[:,y])) == len(s[:,y]):
            r += 10

        for i in range(len(s)):
            for j in range(len(s)):
                if s[i][y] == val or s[x][j] == val:
                    r = -1
                    self.color = self.RED_COLOR

        return r

    def step(self, action, s= True):
        if s:
            s = self.s
        i,j, value = action
        reward = self.reward(i,j,value,s)
        s[i][j] = value
        self.actions.append([i, j, value, self.color])
        
        return self.observation_space(), reward, self.done(s), self.info()
      
class Discrete:
    def __init__(self, s):
        self.action_space = [(x//9, x % 9, i) for x in np.where(np.hstack(s) == 0)[0] for i in range(1,10)]
    
    def __repr__(self):
        return f'Discrete({self.n()})'

    def n(self):
        return len(self.action_space)

    def sample(self):
        return self.action_space[np.random.randint(self.n())]

class Box:

    def __init__(self, s, actions):
        self.actions = actions
        self.s = s
    
    def __call__(self):
        return self.render(self.s)
    
    def __repr__(self):
        return f'Box{self.s.shape}'

env = make('Sudoku-v0')

'''
for i in range(50):
    rnd_action = env.action_space.sample()
    obs, reward, done, info =env.step(rnd_action)
    if done:
        break
'''