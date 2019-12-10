import gym
import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class DRQNetwork:
	def __init__(self, state_size=2, action_size=2, hidden_size=10, learning_rate=0.01,seq_length=5):
		self.state_size = state_size
		self.obs_size = state_size
		self.act_size = action_size
		self.seq_length = seq_length
		# State inputs to the Q-Network
		self.inputs_ = tf.placeholder(tf.float32, [None,seq_length, state_size])

		# One hot encode the actions to later choose the Q value for the action
		self.actions_ = tf.placeholder(tf.int32, [None])
		one_hot_actions = tf.one_hot(self.actions_, action_size)

		# Target Q values for training
		self.Q_target = tf.placeholder(tf.float32, [None])

		# Relu hidden layers
		self.layer1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
		# RNN stuff
		lstm_layer = tf.keras.layers.LSTM(hidden_size, input_shape=(None, hidden_size))(self.layer1)
		self.layer2 = tf.contrib.layers.fully_connected(lstm_layer, hidden_size)
		self.output = tf.contrib.layers.fully_connected(self.layer2, action_size, activation_fn=None)

		self.actSel = tf.argmax(self.output, 1)

		# Train with loss (Q_target - Q)^2
		self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
		self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q))
		self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		self.init = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init)

	def __call__(self, z, memory):
		i = 0
		in_seq = []
		while i < self.seq_length-1:
			t_i = memory.buffer[-1-i]
			if (t_i[3] == np.zeros(self.state_size)).all(): break
			in_seq.append(t_i[0])
			i += 1
		in_seq.reverse()
		in_seq.append(z)
		s = tf.keras.preprocessing.sequence.pad_sequences(np.array(in_seq).T, maxlen=self.seq_length,
													  dtype='float64', padding='pre', truncating='pre', value=0.0).reshape(1,-1,self.state_size)
		pred = self.sess.run(self.actSel, feed_dict={self.inputs_: s})
		return pred[0]



class Memory():
	def __init__(self, max_size=1000):
		self.buffer = deque(maxlen=max_size)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size,seq_length=None):
		index = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
		if seq_length:
			ret_seq = []
			for ii in index:
				max_i = min(ii+seq_length,len(self.buffer))
				ret_seq.append([self.buffer[k_i] for k_i in range(ii,max_i)])
			return ret_seq
		else:
			return [self.buffer[ii] for ii in index]

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / N


def DRQN(env, gamma, num_episodes=100, run=1):

	# Exploration params
	explore_start = 1.0
	explore_stop = 0.01
	decay_rate = 0.0001

	# Network params
	hidden_size = 32
	learning_rate = 3e-4

	# Memory params
	batch_size = 20  # Number of experiences stored in memory when initialized for first time
	seq_length = 5
	memory_size = 10000  # Number of experiences the memory can keep
	pretrain_length = batch_size

	# Video Path (dummy)
	video_path = 'videos/drqn_run_ep_.mp4'

	QN = DRQNetwork(state_size=2,hidden_size=hidden_size, learning_rate=learning_rate,seq_length=seq_length)

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
		if episode == 10 or episode == 100 or episode == 900:
			video_path = 'videos/drqn_run{}_ep{}_.mp4'.format(run, episode)
			video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
			# env.unwrapped.render()
		else:
			video_recorder = VideoRecorder(env, video_path, enabled=False)
		while not done:
			exp_step += 1
			# # Watch sim
			# env.render()

			# Explore or Exploit
			explore_prob = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * exp_step)
			if explore_prob > np.random.rand():
				# Random action
				action = env.action_space.sample()
			else:
				# Action from Q Network
				action = QN(z, memory)

			# Take action
			state_prime, reward, done, _ = env.step(action)
			# capture frame
			video_recorder.capture_frame()
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
			batch = memory.sample(batch_size,seq_length=seq_length)
			zs = np.array([tf.keras.preprocessing.sequence.pad_sequences(np.array([each[0] for each in sub_batch]).T,maxlen=seq_length,dtype='float64', padding='pre', truncating='pre', value=0.0).T for sub_batch in batch])
			# actions = np.array([tf.keras.preprocessing.sequence.pad_sequences(np.array([[each[1]] for each in sub_batch]).T,maxlen=seq_length,dtype='float64', padding='pre', truncating='pre', value=0.0).T for sub_batch in batch])
			actions = np.array([sub_batch[-1][1] for sub_batch in batch])
			rewards = np.array([sub_batch[-1][2] for sub_batch in batch])
			z_primes = np.array([tf.keras.preprocessing.sequence.pad_sequences(np.array([each[3] for each in sub_batch]).T,maxlen=seq_length,dtype='float64', padding='pre', truncating='pre', value=0.0).T for sub_batch in batch])

			# Train Network
			Q_target = QN.sess.run(QN.output, feed_dict={QN.inputs_: z_primes})

			# Set Q_target to 0 for states where episode ends
			episode_ends = (z_primes[:,-1,:] == np.zeros(z[0].shape)).all(axis=1)
			Q_target[episode_ends] = (0, 0)

			targets = rewards + gamma * np.max(Q_target, axis=1)
			loss, _ = QN.sess.run([QN.loss, QN.optimizer],
							   feed_dict={
								   QN.inputs_: zs,
								   QN.Q_target: targets,
								   QN.actions_: actions
							   })
		G_0.append(total_reward)
		video_recorder.close()

	print("Max G_0 {}".format(max(G_0)))
	# QN.sess.close()
	return G_0, QN