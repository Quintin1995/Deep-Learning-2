# The main a3c file

import os
import sys
import threading
import multiprocessing
import gym
from queue import Queue
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .parameters import *
from .network import ActorCriticNetwork
from .worker import Worker
from .utils import logger

# Executes all tf commands in the order we give them, does not try to optimize execution
tf.enable_eager_execution()

class MasterAgent():
	def __init__(self):
		# Disable GPU if one is detected, this is a CPU only task because of multithreading
		os.environ['CUDA_VISIBLE_DEVICES'] = ''
		self.game_name = A_GAME_NAME
		self.save_dir = N_MODEL_FILE_PATH
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		env = gym.make(self.game_name)
		self.state_size = env.observation_space.shape
		#self.state_size = (84,84,4)
		self.state_size = (84,84,A_FRAME_BUFFER) # adaptable dimensions
		self.action_size = env.action_space.n
		self.opt = tf.train.AdamOptimizer(A_LEARN_RATE, use_locking=True)
		self.global_model = ActorCriticNetwork(self.state_size, self.action_size)  # global network
		tensor = tf.convert_to_tensor([np.random.random(self.state_size)], dtype=tf.float32)
		policy, values = self.global_model(tensor)
		self.game_lengths = []
		self.stats_file_name = "saved_models/results.txt"
		self.check_file()

	def check_file(self):
		# removes file if its there so we don't append new results to old stuff
		if os.path.isfile(self.stats_file_name):
			os.remove(self.stats_file_name)

	def append_to_file(self,s):
		with open(self.stats_file_name,'a+') as f:
			f.write(s + "\n")

	def train(self):
		default_cpu_num = multiprocessing.cpu_count()
		if default_cpu_num <= A_MAX_CPU:
			cpu_num = default_cpu_num
		else:
			cpu_num = A_MAX_CPU
		#cpu_num = 1 # limiting to 1 thread for debugging
		res_queue = Queue()
		workers = [Worker(self.state_size,
							self.action_size,
							self.global_model,
							self.opt, res_queue,
							i,
							save_dir=self.save_dir) for i in range(cpu_num)]

		for i, worker in enumerate(workers):
			#print("Starting worker {}".format(i))
			logger.info("Starting worker {}".format(i))
			worker.start()

		moving_average_rewards = []  # record episode reward to plot
		while True:
			reward = res_queue.get()
			#print("Getting from res_queue")
			if reward is not None:
				moving_average_rewards.append(reward)
				self.append_to_file(str(reward))
			elif Worker.global_episode < A_MAX_EPS:
				continue
			else:
				break
		[w.join() for w in workers]
		print(moving_average_rewards)
		plt.plot(moving_average_rewards)
		plt.ylabel('Moving average ep reward')
		plt.xlabel('Step')
		plt.savefig(os.path.join(self.save_dir,'{} Moving Average.png'.format("breakout")))
		#plt.show()

	def play(self):
		env = gym.make(self.game_name).unwrapped
		state = env.reset()
		model = self.global_model
		model_path = os.path.join(self.save_dir, 'model_{}.h5'.format("breakout"))
		print('Loading model from: {}'.format(model_path))
		model.load_weights(model_path)
		done = False
		step_counter = 0
		reward_sum = 0

		try:
			while not done:
				env.render(mode='rgb_array')
				policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
				policy = tf.nn.softmax(policy)
				action = np.argmax(policy)
				state, reward, done, _ = env.step(action)
				reward_sum += reward
				print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
				step_counter += 1
		except KeyboardInterrupt:
			print("Received Keyboard Interrupt. Shutting down.")
		finally:
			env.close()


