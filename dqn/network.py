#Implement Simple CNN and train on character dataset
from dqn.parameters import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Input, Conv2D
import numpy as np, gym, sys, copy, argparse
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential,Model
import random
import tensorflow as tf
from collections import deque
from pathlib import Path
import keras
from keras import backend as K_back
import keras.backend as K
from gym.wrappers import Monitor
import pickle
import os


class QNetwork():

	def __init__(self, env, model_type=None):
		self.learning_rate = 0.0001														#######Hyperparameter
		self.obs_space     = env.observation_space.shape + (-1,)
		self.ac_space      = env.action_space.n
		self.model_type    = model_type
		
		if(model_type=='DQN'):
			print("Building DQN model")
			self.model=self.build_model_DQN()

		elif(model_type=='Dueling' or model_type=='dueling'):
			print("Dueling  Model")
			self.model=self.build_model_dueling()
		elif(model_type=='Dueling2' or model_type=='dueling2'):
			print("Dueling  Model2")
			self.model=self.build_model_dueling_second()
		else:
			print("Incorrect Model")
			assert 0==1
		
	def save_model_weights(self, name):
		# Helper function to save your model / weights. 
		self.model.save(name)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.model=keras.models.load_model(model_file, custom_objects={"K_back": K_back})

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		self.model.load_weights(weight_file)


	#default DQN, baseline model
	def build_model_DQN(self):
		#Builds a DQN
		model=Sequential()
		# model.add(Input(self.obs_space))
		model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last", input_shape=(84, 84, 4)))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last"))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())

		model.add(Dense(units=150,input_dim=self.obs_space,activation='relu',kernel_initializer='he_uniform'))
		model.add(Dense(units=150,activation='relu',kernel_initializer='he_uniform'))
		model.add(Dense(units=self.ac_space,activation='linear',kernel_initializer='he_uniform'))
		# model.build(input_shape=self.obs_space)
		model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
		model.summary()
		return model


	#best dueling model
	def build_model_dueling(self):
		inp = Input(shape=(84, 84, 4))
		x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last")(inp)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last")(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last")(x)
		# x = MaxPooling2D(pool_size=(2, 2))(x)
		
		x = Flatten()(x)

		x = Dense(units=150,activation='relu',kernel_initializer='he_uniform',name='hidden_layer_1')(x)
		x = Dense(units=150,activation='relu',kernel_initializer='he_uniform',name='hidden_layer_2')(x)


		value_=Dense(units=1,activation='linear',kernel_initializer='he_uniform',name='Value_func')(x)
		ac_activation=Dense(units=self.ac_space,activation='linear',kernel_initializer='he_uniform',name='action')(x)
		#Compute average of advantage function
		avg_ac_activation=Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
		#Concatenate value function to add it to the advantage function
		concat_value=Concatenate(axis=-1,name='concat_0')([value_,value_])
		concat_avg_ac=Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])
		
		for i in range(1,self.ac_space-1):
			concat_value=Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
			concat_avg_ac=Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])
		
		#Subtract concatenated average advantage tensor with original advantage function
		ac_activation=Subtract()([ac_activation,concat_avg_ac])
		#Add the two (Value Function and modified advantage function)
		merged_layers=Add(name='final_layer')([concat_value,ac_activation])
		final_model=Model(inputs=inp,outputs=merged_layers)
		final_model.summary()
		final_model.compile(loss='mean_squared_error',optimizer=Adam(lr=self.learning_rate))
		return final_model
	

	def build_model_dueling_second(self):
		inp = Input(shape=(84, 84, 4))
		x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last")(inp)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last")(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format="channels_last")(x)
		# x = MaxPooling2D(pool_size=(2, 2))(x)
		
		x = Flatten()(x)

		x   = Dense(150, activation = 'relu')(x)
		x   = Dense(150, activation = 'relu')(x)
		if(self.model_type == "dueling2"):
			x = Dense(self.ac_space + 1, activation='linear')(x)
			x = Lambda(lambda i: K.expand_dims(i[:,0], -1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.ac_space,))(x)
		x = Dense(self.ac_space, activation='linear')(x)
		return Model(inp,x)