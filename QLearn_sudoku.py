from Sudoku import make
import numpy as np
import sys

def main(alpha=0.9, gamma=0.99, episodes=10000, epsilon=1):

    env = make('Sudoku-v0')

    Q,rpe = np.zeros((env.observation_space.n,env.action_space.n)), list()
    
    for e in range(episodes):

        s, er = env.reset(), 0
        
        if e % 1000 == 999:
            print(f'Trained Episodes:[{e + 1}/{episodes}] --> Mean Rewards per Episode:{np.mean(rpe[e - 999:e]):.3f}')

        for _ in range(120):
            
            if e == episodes - 1:
                
                if len(env.render_buffer) != 0:
                    sys.stdout.write('\x1b[1A')
                    sys.stdout.write('\x1b[2K')
                env.render()
                print(f'Rendering GIF: {len(env.render_buffer)/1.2}%')
      

            if np.random.uniform(0,1) < epsilon:
                
                a = env.action_space.sample()

            else:   

                a = np.argmax(Q[s,:])

            s_, r, done, _ = env.step(a)

            Q[s,a] = alpha*Q[s,a] + (1- alpha)*(r + gamma * max(Q[s_,:]))

            er += r
            
            if done:
                break

            s = s_

        epsilon = max(0.01, np.exp(-0.001*e))
        rpe.append(er)


    env.animate()

if __name__ == '__main__':
    main()
