# Utilities file for actor critic
import logging
import sys

logger = logging.getLogger()

logging.basicConfig(
	stream=sys.stdout,
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.DEBUG,
	datefmt='%Y-%m-%d %H:%M:%S')

class Memory:
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []

	def store(self, state, action, reward):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)

	def clear(self):
		self.states = []
		self.actions = []
		self.rewards = []

def record(episode,
			episode_reward,
			worker_idx,
			global_ep_reward,
			result_queue,
			total_loss,
			num_steps):
	"""Helper function to store score and print statistics.
	Arguments:
		episode: Current episode
		episode_reward: Reward accumulated over the current episode
		worker_idx: Which thread (worker)
		global_ep_reward: The moving average of the global reward
		result_queue: Queue storing the moving average of the scores
		total_loss: The total loss accumualted over the current episode
		num_steps: The number of steps the episode took to complete
	"""
	if global_ep_reward == 0:
		global_ep_reward = episode_reward
	else:
		global_ep_reward = global_ep_reward * 0.0 + episode_reward * 1.0
	"""
	print(
			f"Episode: {episode} | "
			f"Moving Average Reward: {int(global_ep_reward)} | "
			f"Episode Reward: {int(episode_reward)} | "
			f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
			f"Steps: {num_steps} | "
			f"Worker: {worker_idx}"
	)
	"""
	result_queue.put(global_ep_reward)
	#print("Putting result in queue.")
	return global_ep_reward