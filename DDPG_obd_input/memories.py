# define price vector memory and experience replay memory
from .config import *

import numpy as np


class Memory(object):
    def __init__(self):
        self.state = np.zeros(shape=[MEMORY_SIZE, INPUT_DIM])
        self.action = np.zeros(shape=[MEMORY_SIZE, ACTION_DIM])
        self.reward = np.zeros(shape = [MEMORY_SIZE, 1])
        self.state_ = np.zeros(shape=[MEMORY_SIZE, INPUT_DIM])

        self.memory_idx = 0


    def store(self, state, action, reward, state_):

        idx = self.memory_idx % MEMORY_SIZE

        self.state[idx, :] = state
        self.action[idx, :] = action
        self.reward[idx, :] = reward
        self.state_[idx, :] = state_

        self.memory_idx += 1


    def extract_batch(self, batch_size):

        max_idx = min(self.memory_idx, MEMORY_SIZE)
        batch_size = min(self.memory_idx, batch_size)

        samples = np.random.choice(a = max_idx, size = batch_size, replace=False)

        return (self.state[samples, :], self.action[samples, :], self.reward[samples, :], self.state_[samples, :])