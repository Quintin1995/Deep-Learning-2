# The main a3c file

import os
import threading
import multiprocessing
import gym
from queue import Queue
import tensorflow as tf

from .parameters import *
from .network import ActorCriticNetwork

# Disable GPU if one is detected, this is a CPU only task because of multithreading
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Executes all tf commands in the order we give them, does not try to optimize execution
tf.enable_eager_execution()

class MasterAgent():
	def __init__(self):
		self.game_name = "Breakout-v0"
		self.save_dir = N_MODEL_FILE_PATH
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		env = gym.make(self.game_name)
		self.state_size = env.observation_space.shape
		self.action_size = env.action_space.n
		print(self.state_size, self.action_size)
		
		self.opt = tf.train.AdamOptimizer(A_LEARN_RATE, use_locking=True)
		self.global_model = ActorCriticNetwork(self.state_size, self.action_size)  # global network
		self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

	def train(self):
		res_queue = Queue()
		workers = [Worker(self.state_size,
							self.action_size,
							self.global_model,
							self.opt, res_queue,
							i,
							save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

		for i, worker in enumerate(workers):
			print("Starting worker {}".format(i))
			worker.start()

		moving_average_rewards = []  # record episode reward to plot
		while True:
			reward = res_queue.get()
			if reward is not None:
				moving_average_rewards.append(reward)
			else:
				break
		[w.join() for w in workers]

		plt.plot(moving_average_rewards)
		plt.ylabel('Moving average ep reward')
		plt.xlabel('Step')
		plt.savefig(os.path.join(self.save_dir,
									'{} Moving Average.png'.format("traffic")))
		#plt.show()

	def play(self):
		env = gym.make(self.game_name).unwrapped
		state = env.reset()
		model = self.global_model
		model_path = os.path.join(self.save_dir, 'model_{}.h5'.format("traffic"))
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


