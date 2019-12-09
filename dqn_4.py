import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

class DQNetwork:
    def __init__(self, state_size=2, action_size=2, hidden_size=10, learning_rate=0.01, name='DQNetwork'):
        with tf.variable_scope(name):
            # State inputs to the Q-Network
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.Q_target = tf.placeholder(tf.float32, [None], name='target')

            # Set up model for RNN layer
            # model = tf.keras.Sequential()
            # model.add(tf.contrib.layers.fully_connected(hidden_size, input_shape=(2,)))
            # model.add(tf.keras.layers.LSTM(hidden_size))
            # model.add(tf.contrib.layers.fully_connected(hidden_size))
            # # model.add(tf.keras.layers.Embedding(input_dim=hidden_size, output_dim=hidden_size))

            # Relu hidden layers
            self.layer1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            # RNN stuff
            lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
            lstm_outputs, _ = tf.keras.layers.RNN(lstm_cell, self.layer1, dtype=tf.float32)
            # - #
            self.layer3 = tf.contrib.layers.fully_connected(lstm_outputs, hidden_size)
            self. output = tf.contrib.layers.fully_connected(self.layer3, action_size, activation_fn=None)

            # Train with loss (Q_target - Q)^2
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in index]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

if __name__ == "__main__":
    # --------- #

    env = gym.make('CartPole-v0')

    # Training params
    train_episodes = 500
    max_steps = 200
    gamma = 0.99

    # Exploration params
    explore_start = 1.0
    explore_stop = 0.01
    decay_rate = 0.0001

    # Network params
    hidden_size = 32
    learning_rate = 3e-4

    # Memory params
    batch_size = 20 # Number of experiences stored in memory when initialized for first time
    memory_size = 10000  # Number of experiences the memory can keep
    pretrain_length = batch_size

    # Runs
    runs = 1

    tf.reset_default_graph()
    QN = DQNetwork(hidden_size=hidden_size, learning_rate=learning_rate, name='main')

    # --------- #
    cum_rewards_list = []
    epis = [[]]
    for k in range(runs):
        epis[0] = []
        cum_rewards_list.append([])
        # Initialize sim
        env.reset()
        # Random step to start
        state, reward, done, _ = env.step(env.action_space.sample())
        # - # Mask the state
        obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        z = np.matmul(obs_mask, state)
        # - #
        memory = Memory(max_size=memory_size)
        # Pre-train to fill memory
        for i in range(pretrain_length):
            # # Watch sim
            # env.render()

            # Take random action
            action = env.action_space.sample()
            state_prime, reward, done, _ = env.step(action)
            # - # Mask the state prime
            z_prime = np.matmul(obs_mask, state_prime)
            # - #

            if done:
                # Sim is done so no state prime
                state_prime = np.zeros(state_prime.shape)
                # - # Mask the state prime
                z_prime = np.matmul(obs_mask, state_prime)
                # - #
                memory.add((z, action, reward, z_prime))
                # Start new episode
                env.reset()
                # Random step to start
                state, reward, done, _ = env.step(env.action_space.sample())
                # - # Mask the state
                z = np.matmul(obs_mask, state)
                # - #
            else:
                # Add experience to memory
                memory.add((z, action, reward, z_prime))
                z = z_prime

        saver = tf.train.Saver()
        rewards_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            step = 0
            for episode in tqdm(range(1, train_episodes)):
                # for episode in range(1, train_episodes):
                total_reward = 0
                t = 0
                while t < max_steps:
                    step += 1
                    # # Watch sim
                    # env.render()

                    # Explore or Exploit
                    explore_prob = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                    if explore_prob > np.random.rand():
                        # Random action
                        action = env.action_space.sample()
                    else:
                        # Action from Q Network
                        Qs = sess.run(QN.output, feed_dict={QN.inputs_: z.reshape((1, *z.shape))})
                        action = np.argmax(Qs)

                    # Take action
                    state_prime, reward, done, _ = env.step(action)
                    # - # Mask the state prime
                    z_prime = np.matmul(obs_mask, state_prime)
                    # - #
                    total_reward += reward

                    # Store transition
                    if done:
                        # Episode is over so no state prime
                        z_prime = np.zeros(z.shape)
                        t = max_steps

                        # print('Episode: {}'.format(episode),
                        #       'Total Reward: {}'.format(total_reward),
                        #       'Training Loss: {:.4f}'.format(loss),
                        #       'Explore Probability: {:.4f}'.format(explore_prob))
                        rewards_list.append((episode, total_reward))
                        cum_rewards_list[k].append(total_reward)
                        epis[0].append(episode)

                        # Add experience to memory
                        memory.add((z, action, reward, z_prime))
                        # Start new episode
                        env.reset()
                        # Take random step to start
                        state, reward, done, _ = env.step(env.action_space.sample())
                        # - # Mask the state
                        z = np.matmul(obs_mask, state)
                        # - #
                    else:
                        # Add experience to memory
                        memory.add((z, action, reward, z_prime))
                        z = z_prime
                        t += 1

                    # Sample mini-batch from memory
                    batch = memory.sample(batch_size)
                    zs = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    z_primes = np.array([each[3] for each in batch])

                    # Train Network
                    Q_target = sess.run(QN.output, feed_dict={QN.inputs_: z_primes})

                    # Set Q_target to 0 for states where episode ends
                    episode_ends = (z_primes == np.zeros(z[0].shape)).all(axis=1)
                    Q_target[episode_ends] = (0,0)

                    targets = rewards + gamma*np.max(Q_target, axis=1)
                    loss, _ = sess.run([QN.loss, QN.optimizer],
                                       feed_dict={
                                           QN.inputs_: zs,
                                           QN.Q_target: targets,
                                           QN.actions_: actions
                                       })
            # saver.save(sess, "checkpoints/cartpole.ckpt")
        # eps, rews = np.array(rewards_list).T
        # smoothed_rews = running_mean(rews, 10)
        # plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
        # plt.plot(eps, rews, color='grey', alpha=0.3)
        # plt.xlabel('Episode')
        # plt.ylabel('Total Reward')
        # plt.show()

    a = np.asarray(cum_rewards_list)
    mean = np.mean(a, axis=0)
    eps = np.asarray(epis[0])
    plt.plot(eps, mean.T)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()










