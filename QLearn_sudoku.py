import numpy as np
import gym
from Sudoku import make

def main(alpha=0.9, gamma=0.99, episodes=1000, epsilon=1):

    env = make('Sudoku-v0')

    Q,rpe = np.zeros((env.observation_space.n,env.action_space.n)), list()
    
    for e in range(episodes):

        s, er = env.reset(), 0
        
        for i in range(100):
            
            if np.random.uniform(0,1) < epsilon:
                
                a = env.action_space.sample()

            else:   

                a = np.argmax(Q[s,:])

            s_, r, done, _ = env.step(a)

            Q[s,a] = alpha*Q[s,a] + (1- alpha)*(r + gamma * max(Q[s_,:]))

            er += er + r

            if done:
                break
            s = s_

        epsilon = max(0.01, np.exp(-0.001*e))
        rpe.append(er)

        if e % 200 == 199:
            print(f'Trained Episodes:[{e + 1}/{episodes}] --> Mean Rewards per Episode:{np.log(np.mean(rpe[e - 199:e])):.3f}')


if __name__ == '__main__':
    main()
