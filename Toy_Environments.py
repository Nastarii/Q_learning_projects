import numpy as np
import gym

def main(alpha=0.9, gamma=0.99, episodes=10000, epsilon=1):

    env = gym.make('FrozenLake-v1')

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

        if e % 1000 == 999:
            print(f'Trained Episodes:[{e + 1}/{episodes}] --> Mean Rewards per Episode:{np.mean(rpe[e - 999:e]):.3f}')


if __name__ == '__main__':
    main()

    