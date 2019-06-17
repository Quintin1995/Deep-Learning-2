# Utilities file for actor critic

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

def compute_loss(self,
									 done,
									 new_state,
									 memory,
									 gamma=0.99):
		if done:
			reward_sum = 0.  # terminal
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
	return global_ep_reward