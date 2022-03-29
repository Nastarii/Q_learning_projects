from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class makeEnvironment():

    
    s = np.array([[1,5,0,8,0,3,4,0,6],
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

    def __init__(self):
        self.actions = []
        self.ini = self.s.copy()

    def step(self, action):
        
        i,j, value = action
        reward = self.reward(i,j,value)
        self.s[i][j] = value
        self.actions.append([i, j, value, self.color])
        
        return self.observation(), reward, self.done()

    def action_space(self):
        return [(x//9, x % 9, i) for x in np.where(np.hstack(self.s) == 0)[0] for i in range(1,10)]
    
    def done(self):
        for idx in range(len(self.s)):
            if len(np.unique(self.s[idx])) != 9:
                return False
            if len(np.unique(self.s[:,idx])) != 9:
                return False
        return True

    def info(self):
        pass

    def n(self):
        return len(self.action_space())
    
    def observation(self):
        pass

    def render(self):
        fig, ax = plt.subplots()

        fig.patch.set_visible(False)
        fig.canvas.manager.set_window_title('Sudoku Board')
        
        ax.axis('off')
        ax.axis('tight')

        df = pd.DataFrame(self.s)
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

        plt.show()
    
    def reset(self):
        self.s = self.ini

    def reward(self, x, y, val):
        self.color, r = False, 0

        if len(np.unique(self.s[x])) == len(self.s[x]) or \
           len(np.unique(self.s[:,y])) == len(self.s[:,y]):
            r += 10

        for i in range(len(self.s)):
            for j in range(len(self.s)):
                if self.s[i][y] == val or self.s[x][j] == val:
                    r = -1
                    self.color = self.RED_COLOR

        if not self.color:
            r += 1
            self.color = self.GREEN_COLOR

        return r

    def sample(self):
        return self.action_space()[np.random.randint(self.n())]
      


env = makeEnvironment()
#print(env.action_space())
rnd_action = env.sample()
#rnd_action = (8,0,4)
print(rnd_action)
env.step(rnd_action)
env.render()
