import sys
import numpy as np
import gym
from matplotlib import pyplot as plt
from dqn import DQN
from drqn import DRQN, Memory
from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN
from reinforce_Buffer import REINFORCE as RF_Buffer, PiApproximationWithNN as Pi_Buffer, ReplayMemory #, Baseline, VApproximationWithNN
# from MCTS import StateActionFeatureVectorWithTile,MCTS
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def test_DQN(env, run):
    gamma = 1.0
    return DQN(env, gamma, 1000, run)


def test_DRQN(env, run):
    gamma = 1.0
    return DRQN(env, gamma, 1000, run)


def test_MCTS():
    # X = StateActionFeatureVectorWithTile(
    #     env.observation_space.low,
    #     env.observation_space.high,
    #     env.action_space.n,
    #     num_tilings=10,
    #     tile_width=np.array([.45, .035])
    # )
    pass


def test_reinforce(runs, with_baseline):
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

    return REINFORCE(env, gamma, 1000, runs, pi, B)
    

def test_reinforce_Buffer(with_baseline, mem_size, runs):
    env = gym.make("CartPole-v0")
    gamma = 1.
    alpha = 3e-4

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.reset_default_graph()

    pi = Pi_Buffer(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha,
        mem_size)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        B = Baseline(0.)

    return RF_Buffer(env, gamma, 100, runs, pi, B, mem_size)

def play_obs(env,pi,num_episodes=10, video_path=None):
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
    env.close()

def play_with_buffer(env,pi,num_episodes=3,video_path=None):
    video_path = 'videos/REINFORCE_best.mp4'
    video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
    rep = ReplayMemory(pi.config)
    obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    for e_i in range(num_episodes):
        s = env.reset()
        z = np.matmul(obs_mask, s)
        rep.add(z, 0, 0)
        # env.unwrapped.render()
        done = False
        while not done:
            a = int(pi(rep.getState()))
            s_prime, r_t, done, _ = env.step(a)
            z_prime = np.matmul(obs_mask, s_prime)
            rep.add(z_prime, r_t, a)
            video_recorder.capture_frame()
            # env.render()
            # s = s_prime
            # z = z_prime
    video_recorder.close()
    # video_recorder.enabled = False
    env.close()

def play(env,pi,num_episodes=10,video_path=None):
    # video_recorder = VideoRecorder(env,video_path,enabled=video_path is not None)
    # obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    for e_i in range(num_episodes):
        s = env.reset()
        # z = np.matmul(obs_mask,s)
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
    env.close()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

if __name__ == "__main__":
    num_iter = 5
    env = gym.make("CartPole-v0")

    # without_buffer = []
    # for q in range(num_iter):
    #     print('***************************************')
    #     print("----------------> Without Buffer: {}".format(q))
    #     print('***************************************')
    #     training_progress = test_reinforce(q, with_baseline=False)
    #     without_buffer.append(training_progress[0])
    #     pi = training_progress[1]
    # without_buffer = np.mean(without_buffer, axis=0)
    # # play(env,pi)
    #
    # # Test REINFORCE_buffer size 2 and 5
    # with_buffer2 = []
    # for q in range(num_iter):
    #     print('***************************************')
    #     print("----------------> With Buffer = 2: {}".format(q))
    #     print('***************************************')
    #     training_progress = test_reinforce_Buffer(False, 2, q)
    #     with_buffer2.append(training_progress[0])
    #     pi_buff = training_progress[1]
    # with_buffer2 = np.mean(with_buffer2, axis=0)
    # # play_with_buffer(env, pi_buff)

    with_buffer5 = []
    for q in range(num_iter):
        print('***************************************')
        print("----------------> With Buffer = 5: {}".format(q))
        print('***************************************')
        training_progress = test_reinforce_Buffer(False, 5, q)
        with_buffer5.append(training_progress[0])
        pi_buff = training_progress[1]
    with_buffer5 = np.mean(with_buffer5, axis=0)
    env._max_episode_steps = 10000
    play_with_buffer(env, pi_buff)
    #
    # # Plot the experiment result
    # fig,ax = plt.subplots()
    # ax.plot(np.arange(len(without_buffer)), without_buffer, label='No Buffer')
    # ax.plot(np.arange(len(with_buffer2)), with_buffer2, label='Buffer - Size 2')
    # ax.plot(np.arange(len(with_buffer5)), with_buffer5, label='Buffer - Size 5')
    # #
    # ax.set_xlabel('iteration')
    # ax.set_ylabel('G_0')
    # ax.legend()
    #
    # plt.show()

    # ##########################################################################
    # # Test DQN
    # dqn_list = []
    # dqn_policies = []
    # for q in range(num_iter):
    #     dqn_rew, dqn_pi = test_DQN(env, q)
    #     dqn_list.append(dqn_rew)
    #     dqn_policies.append(dqn_pi)
    # dqn_result = np.mean(dqn_list,axis=0)
    # smoothed_dqn_result = running_mean(dqn_result, 10)
    #
    # # Test DRQN
    # drqn_list = []
    # drqn_policies = []
    # for q in range(num_iter):
    #     drqn_rew, drqn_pi = test_DRQN(env, q)
    #     drqn_list.append(drqn_rew)
    #     drqn_policies.append(drqn_pi)
    # drqn_result = np.mean(drqn_list, axis=0)
    # smoothed_drqn_result = running_mean(drqn_result, 10)
    #
    # # # Plot the experiment result
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(smoothed_dqn_result)), smoothed_dqn_result, label='DQN_smoothed')
    # ax.plot(np.arange(len(dqn_result)), dqn_result, label='DQN', color='red', alpha=0.3)
    # ax.plot(np.arange(len(smoothed_drqn_result)), smoothed_drqn_result, label='DRQN_smoothed')
    # ax.plot(np.arange(len(drqn_result)), drqn_result, label='DRQN', color='grey', alpha=0.3)
    #
    # ax.set_xlabel('iteration')
    # ax.set_ylabel('G_0')
    # ax.legend()
    #
    # plt.show()
    # ##########################################################################
    # obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    # batch_size = 20  # Number of experiences stored in memory when initialized for first time
    # memory_size = 10000  # Number of experiences the memory can keep
    #
    # # - - - - #
    # env.reset()
    # state, _, _, _ = env.step(env.action_space.sample())
    # z = np.matmul(obs_mask, state)
    # memory = Memory(max_size=memory_size)
    # # Pre-train to fill memory
    # for i in range(batch_size):
    #     # # Watch sim
    #     # env.render()
    #
    #     # Take random action
    #     action = env.action_space.sample()
    #     state_prime, reward, done, _ = env.step(action)
    #     z_prime = np.matmul(obs_mask, state_prime)
    #
    #     if done:
    #         # Sim is done so no state prime
    #         state_prime = np.zeros(state.shape)
    #         z_prime = np.matmul(obs_mask, state_prime)
    #         memory.add((z, action, reward, z_prime))
    #         # Start new episode
    #         env.reset()
    #         # Random step to start
    #         state, reward, done, _ = env.step(env.action_space.sample())
    #         z = np.matmul(obs_mask, state)
    #     else:
    #         # Add experience to memory
    #         memory.add((z, action, reward, z_prime))
    #         z = z_prime
    # # - - - - #
    #
    # # - - - - #
    # total_reward = 0
    # env._max_episode_steps = 10000
    # env.reset()
    # # memory = Memory(max_size=memory_size)
    # # Take random step to start
    # state, reward, _, _ = env.step(env.action_space.sample())
    # z = np.matmul(obs_mask, state)
    # done = False
    # video_path = 'videos/drqn_best.mp4'
    # video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
    # while not done:
    #     # # Watch sim
    #     # env.render()
    #
    #     # Action from Q Network
    #     action = drqn_policies[0](z, memory)
    #     # Take action
    #     state_prime, reward, done, _ = env.step(action)
    #     z_prime = np.matmul(obs_mask, state_prime)
    #     # capture frame
    #     video_recorder.capture_frame()
    #     total_reward += reward
    #
    #     # Store transition
    #     if done:
    #         # Episode is over so no state prime
    #         state_prime = np.zeros(state.shape)
    #         z_prime = np.matmul(obs_mask, state_prime)
    #         # Add experience to memory
    #         memory.add((z, action, reward, z_prime))
    #
    #     else:
    #         # Add experience to memory
    #         memory.add((z, action, reward, z_prime))
    #         z = z_prime
    # # print('Elapsed steps {}'.format(env._elapsed_steps))
    # video_recorder.close()
    # # - - - - #



    # # Random action
    # action = env.action_space.sample()
    # state, _, done, _ = env.step(action)
    # # Get action
    # action = some_code(state)
    # # Take action
    # state_prime, reward, done, _ = env.step(action)
    # state = state_prime


###### Not sure #####
# # # Plot the experiment result
# fig,ax = plt.subplots()
# ax.plot(np.arange(len(dqn_result)),dqn_result, label='No Buffer')























































