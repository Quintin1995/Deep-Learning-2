# Network for a3c
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from .parameters import *

""" IMPORTANT! Because this inherits from keras.Model, the network needs to be defined
	In the __init__ function so all the layers are registered by the model object.
"""
class ActorCriticNetwork(keras.Model):
	def __init__(self,input_dim,action_size):
		super(ActorCriticNetwork,self).__init__()
		self.input_dim = input_dim  # Tuple describing the image input dimension (X,Y,3)
		self.action_dim = action_dim  # Number of possible actions we can take
		
		# Network input definition
		self.inp = Input(self.input_dim)
		self.conv1 = layers.Conv2D(N_CONV_DIM,kernel_size=3,activation='relu')
		self.maxp1 = layers.MaxPooling2D(N_POOL_DIM)
		self.conv2 = layers.Conv2D(N_CONV_DIM,kernel_size=3,activation='relu')
		self.maxp2 = layers.MaxPooling2D(N_POOL_DIM)
		self.dense1 = layers.Dense(N_DENSE_DIM,activation='relu')
		self.policy_logits = layers.Dense(self.action_size)
		self.dense2 = layers.Dense(N_DENSE_DIM,activation='relu')
		self.values = layers.Dense(1)
	
	def conv_block(self,inp):
		# currently unused, left in for possible testing in the future
		x = layers.Conv2D(N_CONV_DIM,kernel_size=3,activation='relu')(inp)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(N_CONV_DIM,kernel_size=3,activation='relu')(x)
		x = layers.MaxPooling2D(N_POOL_DIM)(x)
		return x

	def call_old(self,inp):
		# One full forward pass of the network
		# Currently unused, left in for possible testing in the future
		conv = self.conv_block(self.inp)
		dense1 = layers.Dense(N_DENSE_DIM,activation='relu')(conv)
		policy_logits = layers.Dense(self.action_size)(dense1)
		dense2 = layers.Dense(N_DENSE_DIM,activation='relu')(policy_logits)
		values = layers.Dense(1)(dense2)
		return policy_logits, values

	def __call__(self,inp):
		# One forward pass of the network
		x = self.inp(inp)
		x = self.conv1(x)
		x = self.maxp1(x)
		x = self.conv2(x)
		x = self.maxp2(x)
		x = self.dense1(x)
		policy = self.policy_logits(x)
		x = self.dense2(policy)
		values = self.values(x)
		return policy, values



