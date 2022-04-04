from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import glob
import os

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

        self.buffer = list()
        self.s = self.rst.copy()
        self.action_space = Discrete()
        self.observation_space = Box()
        self.agent_i, self.agent_j = 0,2
        self.iter = 0

    def done(self,s):
        if np.count_nonzero(s==0) == 0:
            return True
        else:
            return False

    def info(self,s, i,j):
        return {'possibilities': np.count_nonzero(s==0)*9,
                'probability':(np.count_nonzero(s[i] != 0)/9) *  (np.count_nonzero(s[:,j] !=0)/9),
                'iteration':len(self.buffer),
                'maximum repetitions': 6}

    def render(self,s=True):
        if s:
            s = self.s
        
        fig, ax = plt.subplots()

        fig.patch.set_visible(False)
        fig.tight_layout()
        
        
        ax.axis('off')
        ax.axis('tight')


        l = 0.1096
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((l/3*i -0.0548, l/3*j -0.0548),l/3,l/3, facecolor='none', edgecolor='black', linewidth=2))
        
        df = pd.DataFrame(s)
        df[df == 0] = ''
        table = ax.table(cellText=df.values, cellLoc='center', 
                 loc='center', bbox=[0,0,1,1])
        
        for act in self.buffer:
            if act[3] is not None:
                table.get_celld()[(act[0],act[1])]._text.set_color(act[3])

        table.get_celld()[(self.agent_i,self.agent_j)].set_facecolor('#C4A8FB')
        
        if self.iter < 10:
            n = '0' + str(self.iter)
        else:
            n = str(self.iter)

        plt.savefig('render_outputs/frame_' + n + '_.png')
        plt.close(fig)

        self.iter += 1
    
    def animate(self, imgs= []):
        for img in sorted(glob.glob('render_outputs/*.png')):
            imgs.append(imageio.imread(img))
            os.remove(img)
        
        imageio.mimsave('render_outputs/' + self.name + '.gif', imgs,'GIF',duration= 0.7)


    def reset(self):
        self.s, self.buffer, self.agent_i, self.agent_j = self.rst.copy(), [], 0, 2
        return 4

    def reward(self, x, y, val, s):
        self.color, r = self.GREEN_COLOR, 0.01

        if len(np.unique(s[x])) == len(s[x]) or \
           len(np.unique(s[:,y])) == len(s[:,y]):
            r = 0.1
 
        for i in range(len(s)):
            for j in range(len(s)):
                if s[i][y] == val or s[x][j] == val:
                    r = 0
                    self.color = self.RED_COLOR

        return r

    def to_coordinate(self, position):

        if position == 0 and self.agent_i != 0:     
            self.agent_i -= 1
        elif position == 1 and self.agent_j != 0:   
            self.agent_j -= 1
        elif position == 2 and self.agent_i != 8:   
            self.agent_i += 1
        elif position == 3 and self.agent_j != 8:   
            self.agent_j += 1
            
        return self.agent_i,self.agent_j

    def step(self, action, s= True):
        if s:
            s = self.s

        value, position =self.action_space(action)
        i, j = self.to_coordinate(position)
        
        if value != 0 and s[i][j] == 0:
            
            reward = self.reward(i,j,value,s)
            s[i][j] = value

        else:

            self.color = None
            reward = 0
        
        self.buffer.append([i, j, value, self.color])
        self.observation_space = Box()
        return i*9 + j, reward, self.done(s), self.info(s,i,j)
      
class Discrete:
    def __init__(self):
        self.actions = [(x, y) for x in range(10) for y in range(5)]
        #self.action_space = [(x//9, x % 9, i) for x in np.where(np.hstack(s) == 0)[0] for i in range(1,10)]
        self.n = len(self.actions)

    def __repr__(self):
        return f'Discrete({self.n})'

    def __call__(self,idx):
        return self.actions[idx]

    def sample(self):
        return np.random.randint(self.n)
        
class Box:

    def __init__(self):
        self.high = np.array([9]*3)
        self.low = np.array([0]*3)
        self.n = 81

    def __repr__(self):
        return f'Box(9,9)'

#for i in range(50):
#    rnd_action = env.action_space.sample()
#    obs, reward, done, info =env.step(rnd_action)
#    if done:
#        break