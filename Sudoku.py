from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
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

    def render(self):
        fig, ax = plt.subplots()

        fig.patch.set_visible(False)
        fig.canvas.manager.set_window_title('Sudoku Board')
        
        ax.axis('off')
        ax.axis('tight')

        df = pd.DataFrame(self.s)
        df[df == 0] = ''
        ax.table(cellText=df.values, cellLoc='center', 
                 loc='center', bbox=[0,0,1,1])
        
        l = 0.1096
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((l/3*i -0.0548, l/3*j -0.0548),l/3,l/3, facecolor='none', edgecolor='black', linewidth=2))
        
        fig.tight_layout()

        plt.show()

    def observation(self):
        pass


env = makeEnvironment()
#print(env.action_space())
#print(env.sample())
env.render()
