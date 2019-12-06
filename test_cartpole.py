import sys
import numpy as np
import gym
from matplotlib import pyplot as plt
# from gym.wrappers import VideoRecorder
from dqn import DQN
from drqn import DRQN

def test_DQN():
    env = gym.make("CartPole-v0")
    gamma = 1.
    alpha = 3e-4

    
    return DQN(env,gamma,1000,pi)

def test_DRQN():
    
    return DRQN(env,gamma,1000,pi)

from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN

def test_reinforce(with_baseline):
    env = gym.make("CartPole-v0")
    gamma = 1.
    alpha = 3e-4

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.reset_default_graph()

    pi = PiApproximationWithNN(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        B = Baseline(0.)

    return REINFORCE(env,gamma,250,pi,B)

def play(env,pi,num_episodes=10,video_path=None):
    # video_recorder = VideoRecorder(env,video_path,enabled=video_path is not None)
    for e_i in range(num_episodes):
        s = env.reset()
        env.unwrapped.render()
        # video_recorder.capture_frame()
        done = False
        while not done:
            a = pi(s)
            s_prime, r_t, done, _ = env.step(a)
            env.render()
            s = s_prime
    # video_recorder.close()
    # video_recorder.enabled = False

if __name__ == "__main__":
    num_iter = 1
    env = gym.make("CartPole-v0")
    
    # Test REINFORCE
    without_baseline = []
    for q in range(num_iter):
        training_progress = test_reinforce(with_baseline=False)
        without_baseline.append(training_progress[0])
        pi = training_progress[1]
        
    play(env,pi)

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
