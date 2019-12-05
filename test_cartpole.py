import sys
import numpy as np
import gym
from matplotlib import pyplot as plt
from dqn import DQN
from drqn import DRQN

def test_DQN():
    env = gym.make("CartPole-v0")
    gamma = 1.
    alpha = 3e-4

    
    return DQN(env,gamma,1000,pi)

def test_DRQN():
    
    return DRQN(env,gamma,1000,pi)

def play(env,num_episodes=10,pi):
    for e_i in range(num_episodes):
        s = env.reset()
        env.render()
        done = False
        while not done:
            a = pi(s)
            s_prime, r_t, done, _ = env.step(a)
            env.render()
            s = s_prime


if __name__ == "__main__":
    num_iter = 1
    env = gym.make("CartPole-v0")

    # Test DQN


    # Test DRQN


    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(dqn_output)),dqn_output, label='DQN')
    ax.plot(np.arange(len(drqn_output)),drqn_output, label='DRQN')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()
