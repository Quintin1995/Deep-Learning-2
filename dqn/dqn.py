import gym
from dqn.network import DENSENET
from dqn.agent import Agent
from dqn.parameters import *
import numpy as np
import os
import matplotlib.pyplot as plt

class DQN():
    def __init__(self):
        print("Setting up Lunar Lander environment.")
        self.env = gym.make("LunarLander-v2")
        self.num_act = self.env.action_space.n
        self.num_obs = self.env.observation_space.shape[0]
        self.q_agent = Agent(self.num_act,self.num_obs)
        # Create save folder for plots if it does not yet exist
        if not os.path.exists(R_PLOTS_PATH):
            os.mkdir(R_PLOTS_PATH)
        if not os.path.exists(N_MODEL_FILE_PATH):
            os.mkdir(N_MODEL_FILE_PATH)
    def run_experiment(self):
            amount_actions = self.num_act
            num_observations_state = self.num_obs
            self.q_agent = self.q_agent
            reward_list = []
            avg_reward_list = []
            #loop over each game and reset the game environment
            for game_iterator in range(M_NUM_GAMES+1):
                is_game_done = False
                tot_reward, reward = 0,0
                state = self.env.reset()
                state = np.reshape(state, [1, num_observations_state])
                for frame_iterator in range(M_MAX_FRAMES_PER_GAME):
                    #renders the game if set to true
                    if M_DO_RENDER_GAME and not (game_iterator % M_RENDER_GAME_MODULO):
                        self.env.render()
                    action = self.q_agent.perform_action(state)
                    next_state, reward, is_game_done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, num_observations_state])
                    state      = np.reshape(state, [1, num_observations_state])

                    self.q_agent.store_in_memory(state, action, reward, next_state, is_game_done)
                    state = next_state
                    if is_game_done:
                        break
                    if len(self.q_agent.state_list) > M_PLAY_BATCH_SIZE:
                        self.q_agent.replay_from_memory(M_PLAY_BATCH_SIZE, num_observations_state)
                    tot_reward += reward
                print("Game Number: {} of {}, Exploration: {:.2}, Frames Used: {}, Reward: {:.4}".format(game_iterator, M_NUM_GAMES, self.q_agent.epsilon_max, frame_iterator, tot_reward))
                self.q_agent.update_epsilon_greedy()
                reward_list.append(tot_reward)
                if (game_iterator+1) % R_AVG_RANGE == 0:
                    avg_reward = np.mean(reward_list)
                    avg_reward_list.append(avg_reward)
                    reward_list = []
                    self.plot_results(avg_reward_list)
            self.env.close()
            self.q_agent.model.save(N_MODEL_FILE_PATH + N_MODEL_FILE_NAME)

    def plot_results(self,avg_reward_list,name=R_PLOTS_FILE):
        # Plot Rewards
        plt.plot(10*(np.arange(len(avg_reward_list)) + 1), avg_reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.savefig(R_PLOTS_PATH+name)
        plt.close()


