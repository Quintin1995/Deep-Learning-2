import threading
import os
import multiprocessing  # may be uneccessary, consider a test run with this removed
import gym
import tensorflow as tf
import numpy as np
import cv2
import sys


from .network import ActorCriticNetwork
from .utils import Memory, record, logger
from .parameters import *


# Worker object file
class Worker(threading.Thread):
	# Set up global variables across different threads
	global_episode = 0
	# Moving average reward
	global_moving_average_reward = 0
	best_score = 0
	save_lock = threading.Lock()

	def __init__(self,
					state_size,
					action_size,
					global_model,
					opt,
					result_queue,
					idx,
					save_dir='/tmp'):
		super(Worker, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		self.result_queue = result_queue
		self.global_model = global_model
		self.opt = opt
		self.local_model = ActorCriticNetwork(self.state_size, self.action_size)
		self.worker_idx = idx
		#self.local_ep = 0
		#self.max_q_size = 100000
		#self.traffic_map = Map(self.max_q_size)
		self.env = gym.make(A_GAME_NAME)
		self.save_dir = save_dir
		self.ep_loss = 0.0

	def run(self):

		total_step = 1
		mem = Memory()
		while Worker.global_episode < A_MAX_EPS:
			#self.local_ep = Worker.global_episode
			#print("Epoch: {0}".format(Worker.global_episode))
			logger.info("Game: {0}".format(Worker.global_episode))
			current_state = self.env.reset()
			current_state, _ , _ = self.collect_states(action=0) # trying this out to see if better performance
			mem.clear()
			ep_reward = 0.
			ep_steps = 0
			self.ep_loss = 0
			time_count = 0
			done = False
			n_time_steps = A_MAX_FRAMES # import from parameters so I don't have to edit 3 different files to make something work
			for t in range(0, n_time_steps):
				current_state = [current_state] # adds the extra dimension we expect here
				logits, _ = self.local_model(
								tf.convert_to_tensor(current_state,
															 dtype=tf.float32))
				probs = tf.nn.softmax(logits)
				#print("Worker {}: probabilities {}".format(self.worker_idx,probs))
				action = np.random.choice(self.action_size, p=probs.numpy()[0])
				#print(action)
				# print("ACTION: ", action)
				#new_state, reward, done, _ = self.env.step(action)
				new_state, reward, done = self.collect_states(action=action) # also trying this out here
				# print("REWARD: ", reward)
				ep_reward += reward
				# print("{0} cars are still in system".format((self.env).number_of_cars()))
				mem.store(current_state, action, reward)

				if time_count == A_UPDATE_FREQ or done:
					# Calculate gradient wrt to local model. We do so by tracking the
					# variables involved in computing the loss by using tf.GradientTape
					with tf.GradientTape() as tape:
						total_loss = self.compute_loss(done,
														new_state,
														mem,
														A_GAMMA)
					self.ep_loss += total_loss
					# Calculate local gradients
					grads = tape.gradient(total_loss, self.local_model.trainable_weights)
					# Push local gradients to global model
					self.opt.apply_gradients(zip(grads,self.global_model.trainable_weights))
					# Update local model with new weights
					self.local_model.set_weights(self.global_model.get_weights())
					mem.clear()
					time_count = 0
				#else:
				#  self.env.step(action, t)
				#if done:  # done and print information
				if done:
					#print("Worker {} done at game step {}".format(self.worker_idx,t))
					logger.info("Game {} done at time step {}".format(Worker.global_episode,t))
					break
				ep_steps += 1
				time_count += 1
				current_state = new_state
				total_step += 1

			Worker.global_moving_average_reward = \
				record(Worker.global_episode, ep_reward, self.worker_idx,
							 Worker.global_moving_average_reward, self.result_queue,
							 self.ep_loss, ep_steps)
			# We must use a lock to save our model and to print to prevent data races.
			if ep_reward > Worker.best_score:
				with Worker.save_lock:
					#print("Saving best model to {}, "
					#			"episode score: {}".format(self.save_dir, ep_reward))
					logger.info("Saving best model to {}, episode score: {}".format(
						self.save_dir,
						ep_reward ))
					self.global_model.save_weights(
							os.path.join(self.save_dir,
													 'model_a3c_{}.h5'.format("breakout"))
					)
					Worker.best_score = ep_reward
			Worker.global_episode += 1
			#print("Worker {}, episode reward {}".format(self.worker_idx,ep_reward))
			
			# print("Car stats: ", n_cars[0], n_cars[1], self.env.number_of_cars())
			#print("{0} cars have disappeared".format(n_cars[0] - n_cars[1] - self.env.number_of_cars()))
			self.result_queue.put(None)

############################
# State collection methods
############################
	def collect_states(self, action):
		#all_states = np.zeros((84,84,4), dtype=np.float16)
		all_states = np.zeros((84,84,A_FRAME_BUFFER), dtype=np.float16) # trying with one less frame
		is_game_done = False
		for i in range(A_FRAME_BUFFER): # normal = 4
			if not is_game_done:
				next_state, reward, is_game_done, _ = self.env.step(action)
			all_states[:,:,i] = self.state_reshape(next_state)
		o = list()
		o.append(all_states / 255.0)
		return np.squeeze(np.asarray(o),axis=0), reward, is_game_done

	def state_reshape(self, state):
		return cv2.resize(self.to_greyscale(state), (84, 110))[26:, :]
		
	def to_greyscale(self, img):
		return np.mean(img, axis=2).astype(np.uint8)


###########################
# Loss
###########################
	def compute_loss(self,
						done,
						new_state,
						memory,
						gamma=A_GAMMA):
			if done:
				reward_sum = 0  # terminal
			else:
				reward_sum = self.local_model(
						tf.convert_to_tensor(new_state[None, :],
																 dtype=tf.float32))[-1].numpy()[0]

			# Get discounted rewards
			discounted_rewards = []
			for reward in memory.rewards[::-1]:  # reverse buffer r
				reward_sum = reward + gamma * reward_sum
				discounted_rewards.append(reward_sum)
			discounted_rewards.reverse()

			logits, values = self.local_model(
					tf.convert_to_tensor(np.vstack(memory.states),
															 dtype=tf.float32))
			# Get our advantages
			advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
															dtype=tf.float32) - values
			# Value loss
			value_loss = advantage ** 2

			# Calculate our policy loss
			policy = tf.nn.softmax(logits)
			entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

			policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
																			logits=logits)
			policy_loss *= tf.stop_gradient(advantage)
			policy_loss -= 0.01 * entropy
			total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
			return total_loss