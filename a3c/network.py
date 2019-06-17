# Network for a3c
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from .parameters import *

class ActorCriticNetwork(keras.Model):
	def __init__(self,input_dim,action_size):
		super(ActorCriticNetwork,self).__init__()
		self.input_dim = input_dim  # Tuple describing the image input dimension (X,Y,3)
		self.action_dim = action_dim  # Number of possible actions we can take
		
		# Network input definition
		self.inp = Input(self.input_dim)
	
	def conv_block(self,inp):
		x = layers.Conv2D(N_CONV_DIM,kernel_size=3,activation='relu')(inp)
		x = layers.BatchNormalization()(x)
		x = layers.Conv2D(N_CONV_DIM,kernel_size=3,activation='relu')(x)
		x = layers.MaxPooling2D(N_POOL_DIM)(x)
		return x

	def call(self,inp):
		# One full forward pass of the network
		conv = self.conv_block(self.inp)
		dense1 = layers.Dense(N_DENSE_DIM,activation='relu')(conv)
		policy_logits = layers.Dense(self.action_size)(dense1)
		dense2 = layers.Dense(N_DENSE_DIM,activation='relu')(policy_logits)
		values = layers.Dense(1)(dense2)
		return policy_logits, values



