import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym import Wrapper
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class DQNetwork:
	def __init__(self, state_size=4, action_size=2, hidden_size=10, learning_rate=0.01):
		self.state_size = state_size
		self.obs_size = state_size
		self.act_size = action_size
		# State inputs to the Q-Network
		self.inputs_ = tf.placeholder(tf.float32, [None, state_size])

		# One hot encode the actions to later choose the Q value for the action
		self.actions_ = tf.placeholder(tf.int32, [None])
		one_hot_actions = tf.one_hot(self.actions_, action_size)

		# Target Q values for training
		self.Q_target = tf.placeholder(tf.float32, [None])

		# Relu hidden layers
		self.layer1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
		self.layer2 = tf.contrib.layers.fully_connected(self.layer1, hidden_size)
		self.output = tf.contrib.layers.fully_connected(self.layer2, action_size, activation_fn=None)

		self.actSel = tf.argmax(self.output,1)

		# Train with loss (Q_target - Q)^2
		self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
		self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q))
		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		self.init = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init)

	def __call__(self, s):
		pred = self.sess.run(self.actSel, feed_dict={self.inputs_: s.reshape(-1, self.state_size)})
		return pred[0]



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


def DQN(env,gamma,num_episodes=100):

	# Exploration params
	explore_start = 1.0
	explore_stop = 0.01
	decay_rate = 0.0001

	# Network params
	hidden_size = 32
	learning_rate = 3e-4

	# Memory params
	batch_size = 20  # Number of experiences stored in memory when initialized for first time
	memory_size = 10000  # Number of experiences the memory can keep
	pretrain_length = batch_size

	# Video Path
	video_path = 'videos/dqn_run_ep_.mp4'
	# videos = []
	# video_recorder_1 = None
	# video_recorder_1_ = VideoRecorder(env, video_path, enabled=video_path is not None)
	# video_recorder_2 = None
	# video_recorder_2_ = VideoRecorder(env, video_path, enabled=video_path is not None)
	# video_recorder_3 = None
	# video_recorder_3_ = VideoRecorder(env, video_path, enabled=video_path is not None)

	QN = DQNetwork(state_size=2,hidden_size=hidden_size, learning_rate=learning_rate)

	obs_mask = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
	env.reset()
	# Random step to start
	state, reward, done, _ = env.step(env.action_space.sample())
	z = np.matmul(obs_mask, state)
	memory = Memory(max_size=memory_size)
	# Pre-train to fill memory
	for i in range(pretrain_length):
		# # Watch sim
		# env.render()

		# Take random action
		action = env.action_space.sample()
		state_prime, reward, done, _ = env.step(action)
		z_prime = np.matmul(obs_mask, state_prime)

		if done:
			# Sim is done so no state prime
			state_prime = np.zeros(state.shape)
			z_prime = np.matmul(obs_mask, state_prime)
			memory.add((z, action, reward, z_prime))
			# Start new episode
			env.reset()
			# Random step to start
			state, reward, done, _ = env.step(env.action_space.sample())
			z = np.matmul(obs_mask, state)
		else:
			# Add experience to memory
			memory.add((z, action, reward, z_prime))
			z = z_prime

	exp_step = 0
	G_0 = []
	for episode in tqdm(range(num_episodes)):
		# Start new episode
		env.reset()
		# Take random step to start
		state, reward, done, _ = env.step(env.action_space.sample())
		z = np.matmul(obs_mask, state)
		total_reward = 0
		t = 0
		done = False
		while not done:
			exp_step += 1
			# # Watch sim
			if episode == 10:
				video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
				# env.unwrapped.render()
				video_recorder.capture_frame()
				video_recorder.close()

			# Explore or Exploit
			explore_prob = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * exp_step)
			if explore_prob > np.random.rand():
				# Random action
				action = env.action_space.sample()
			else:
				# Action from Q Network
				action = QN(state)

			# Take action
			state_prime, reward, done, _ = env.step(action)
			z_prime = np.matmul(obs_mask, state_prime)
			total_reward += reward

			# Store transition
			if done:
				# Episode is over so no state prime
				state_prime = np.zeros(state.shape)
				z_prime = np.matmul(obs_mask, state_prime)

				# print('Episode: {}'.format(episode),
				#       'Total Reward: {}'.format(total_reward),
				#       'Training Loss: {:.4f}'.format(loss),
				#       'Explore Probability: {:.4f}'.format(explore_prob))
				# rewards_list.append((episode, total_reward))
				# cum_rewards_list[k].append(total_reward)
				# epis[0].append(episode)

				# Add experience to memory
				memory.add((z, action, reward, z_prime))

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
			Q_target = QN.sess.run(QN.output, feed_dict={QN.inputs_: z_primes})

			# Set Q_target to 0 for states where episode ends
			episode_ends = (z_primes == np.zeros(state[0].shape)).all(axis=1)
			Q_target[episode_ends] = (0, 0)

			targets = rewards + gamma * np.max(Q_target, axis=1)
			loss, _ = QN.sess.run([QN.loss, QN.optimizer],
							   feed_dict={
								   QN.inputs_: zs,
								   QN.Q_target: targets,
								   QN.actions_: actions
							   })
		G_0.append(total_reward)

	print("Max G_0 {}".format(max(G_0)))
	QN.sess.close()
	return G_0, QN