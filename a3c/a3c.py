# The main a3c file

import os
import threading
import multiprocessing
import gym
from random import choice, randint
from queue import Queue
import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import layers

# Disable GPU if one is detected, this is a CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# find out what this does?
tf.enable_eager_execution()

class A3C:
	def __init__(self,act_dim,env_dim,lr=0.0001,gamma=0.99):
		self.act_dim = act_dim  # Number of actions that can be taken
		self.env_dim = env_dim  # Number of possible observable states
		self.lr = lr  # Learning rate
		self.gamma = gamma
		self.global_net = self.buildGlobal()
		self.actor = # How do we make actor object?
		self.critic = # How do we build this one?
