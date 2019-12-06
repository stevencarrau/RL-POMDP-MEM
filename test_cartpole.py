import sys
import numpy as np
import gym
from matplotlib import pyplot as plt
# from gym.wrappers import VideoRecorder
from dqn import DQN
from drqn import DRQN
from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN
from reinforce_Buffer import REINFORCE as RF_Buffer, PiApproximationWithNN as Pi_Buffer, ReplayMemory #, Baseline, VApproximationWithNN
# from MCTS import StateActionFeatureVectorWithTile,MCTS

#
def test_DQN():
    env = gym.make("CartPole-v0")
    gamma = 1.
    alpha = 3e-4

    
    return DQN(env,gamma,1000,pi)

def test_DRQN():
    
    return DRQN(env,gamma,1000,pi)


def test_MCTS():
    # X = StateActionFeatureVectorWithTile(
    #     env.observation_space.low,
    #     env.observation_space.high,
    #     env.action_space.n,
    #     num_tilings=10,
    #     tile_width=np.array([.45, .035])
    # )
    pass

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

    return REINFORCE(env,gamma,1000,pi,B)
    

def test_reinforce_Buffer(with_baseline):
    env = gym.make("CartPole-v0")
    gamma = 1.
    alpha = 3e-4

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.reset_default_graph()

    pi = Pi_Buffer(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        B = Baseline(0.)

    return RF_Buffer(env,gamma,1000,pi,B)

def play(env,pi,num_episodes=10,video_path=None):
    # video_recorder = VideoRecorder(env,video_path,enabled=video_path is not None)
    obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    for e_i in range(num_episodes):
        s = env.reset()
        z = np.matmul(obs_mask,s)
        env.unwrapped.render()
        # video_recorder.capture_frame()
        done = False
        while not done:
            a = pi(z)
            s_prime, r_t, done, _ = env.step(a)
            z_prime = np.matmul(obs_mask, s_prime)
            env.render()
            s = s_prime
            z = z_prime
    # video_recorder.close()
    # video_recorder.enabled = False

def play_with_buffer(env,pi,num_episodes=10,video_path=None):
    # video_recorder = VideoRecorder(env,video_path,enabled=video_path is not None)
    rep = ReplayMemory(pi.config)
    obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    for e_i in range(num_episodes):
        s = env.reset()
        z = np.matmul(obs_mask,s)
        rep.add(z,0,0)
        env.unwrapped.render()
        # video_recorder.capture_frame()
        done = False
        while not done:
            a = pi(rep.getState(rep.current))
            s_prime, r_t, done, _ = env.step(a)
            z_prime = np.matmul(obs_mask, s_prime)
            rep.add(z_prime,r_t,a)
            env.render()
            s = s_prime
            z = z_prime
    # video_recorder.close()
    # video_recorder.enabled = False
    


if __name__ == "__main__":
    num_iter = 5
    env = gym.make("CartPole-v0")

    without_buffer = []
    for q in range(num_iter):
        training_progress = test_reinforce(with_baseline=True)
        without_buffer.append(training_progress[0])
        pi = training_progress[1]
    without_buffer = np.mean(without_buffer,axis=0)
    play(env,pi)


    # Test REINFORCE_buffer
    with_buffer = []
    for q in range(num_iter):
        training_progress = test_reinforce_Buffer(with_baseline=True)
        with_buffer.append(training_progress[0])
        pi_buff = training_progress[1]
    with_buffer = np.mean(with_buffer,axis=0)
    play_with_buffer(env,pi_buff)

    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(without_buffer)),without_buffer, label='No Buffer')
    ax.plot(np.arange(len(with_buffer)),with_buffer, label='Buffer')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()

    # Test DQN


    # Test DRQN


    # Plot the experiment result
    # fig,ax = plt.subplots()
    # ax.plot(np.arange(len(dqn_output)),dqn_output, label='DQN')
    # ax.plot(np.arange(len(drqn_output)),drqn_output, label='DRQN')
    #
    # ax.set_xlabel('iteration')
    # ax.set_ylabel('G_0')
    # ax.legend()
    #
    # plt.show()
