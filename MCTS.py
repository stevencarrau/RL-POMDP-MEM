import numpy as np


class StateActionFeatureVectorWithTile():
	def __init__(self,
	             state_low: np.array,
	             state_high: np.array,
	             num_actions: int,
	             num_tilings: int,
	             tile_width: np.array):
		"""
		state_low: possible minimum value for each dimension in state
		state_high: possible maimum value for each dimension in state
		num_actions: the number of possible actions
		num_tilings: # tilings
		tile_width: tile width for each dimension
		"""
		# TODO: implement here
		self.offset = np.divide(tile_width, num_tilings)
		t_shape = np.array(np.ceil(np.divide(state_high - state_low, tile_width)) + 1, dtype=int)
		self.num_tiles = np.prod(t_shape)
		self.num_tilings = num_tilings
		self.num_actions = num_actions
		self.tile_loc = [state_low - i / num_tilings * tile_width for i in range(num_tilings)]
		self.tile_width = tile_width
		self.indices = np.arange(self.feature_vector_len()).reshape(tuple(t_shape) + (num_tilings, num_actions))
		self.feature_vector = np.ones(self.indices.flatten().shape, dtype=None)
		self.state_low = state_low
		self.state_high = state_high
	
	def feature_vector_len(self) -> int:
		"""
		return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
		"""
		# TODO: implement this method
		return self.num_actions * self.num_tilings * self.num_tiles
	
	def __call__(self, s, done, a) -> np.array:
		"""
		implement function x: S+ x A -> [0,1]^d
		if done is True, then return 0^d
		"""
		# TODO: implement this method
		if done: return np.zeros(self.feature_vector_len())
		s_basis = np.zeros(self.feature_vector_len())
		for i, t_i in enumerate(self.tile_loc):
			for j in range(self.num_actions):
				if a == j:
					tile = np.array(np.floor(np.divide(s - t_i, self.tile_width)), dtype=int)
					s_basis[self.indices[tuple(tile) + (i, j)]] = 1
		return s_basis

class MCTSNode(object):
	def __init__(self):

def MCTS():
	pass

