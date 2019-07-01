import gym
from dqn.network import DENSENET
import cv2
from dqn.agent import Agent
from dqn.parameters import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ON = OKGREEN + "[ON]" + ENDC
    OFF = FAIL + "[OFF]" + ENDC

def onoroff(boolean):
    if boolean:
        return bcolors.ON
    else:
        return bcolors.OFF

class DQN():
    def __init__(self, dueling=True, use_target_network=True, epochs=5000, memory=10000, replay_batch_size=32, replay_modulo=5):
        print("Setting up Lunar Lander environment.")
        self.dueling = dueling
        self.use_target_network = use_target_network
        self.env = gym.make("Breakout-v4")
        self.num_act = self.env.action_space.n
        self.num_obs = self.env.observation_space.shape[0]
        self.epochs = epochs
        self.memory = memory
        self.replay_batch_size = replay_batch_size
        self.replay_modulo = replay_modulo

        self.print_overview()

        if dueling: 
            mdl_type = 'dueling'
        else:
            mdl_type = 'DQN'
        self.q_agent = Agent(self.num_act,self.num_obs, mdl_type, self.env, use_target=use_target_network, memory=self.memory)
        # Create save folder for plots if it does not yet exist
        if not os.path.exists(R_PLOTS_PATH):
            os.mkdir(R_PLOTS_PATH)
        if not os.path.exists(N_MODEL_FILE_PATH):
            os.mkdir(N_MODEL_FILE_PATH)

    def print_overview (self):
        print (bcolors.OKBLUE + "CURRENT SETTINGS" + bcolors.ENDC)
        print ("DUELING:        " + onoroff(self.dueling))
        print ("TARGET NETWORK: " + onoroff(self.use_target_network))
        print ("EPOCHS:         " + str(self.epochs))
        print ("REPLAY MEM.:    " + str(self.memory))
        print ("REPLAY B. SIZE: " + str(self.replay_batch_size))
        print ("REPLAY MODULO:  " + str(self.replay_modulo))
        print ("ACTION SPACE:   " + str(self.num_act))
        print ("OBSERV. SPACE:  " + str(self.num_obs))
        time.sleep(4)

    # Plots the average reward
    def plot_results(self, avg_reward_list, name=R_PLOTS_FILE):
        # Plot Rewards
        plt.plot(10*(np.arange(len(avg_reward_list)) + 1), avg_reward_list)
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward vs Episodes')
        plt.savefig(R_PLOTS_PATH+name)     
        plt.close()


    def to_greyscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    #reshapes the state into 84 by 84 pixels.
    def state_reshape(self, state):
        return cv2.resize(self.to_greyscale(state), (84, 110))[26:, :]

    #concatenate four frames into one frame stack state
    def collect_states(self, action):
        self.env.render()
        all_states = np.zeros((84,84,4), dtype=np.float16)
        is_game_done = False
        for i in range(4):
            if not is_game_done:
                next_state, reward, is_game_done, _ = self.env.step(action)
            all_states[:,:,i] = self.state_reshape(next_state)
        o = list()
        o.append(all_states / 255.0)
        return np.asarray(o), reward, is_game_done

    #run a q-learning experiment
    def run_experiment(self):
        num_observations_state = self.num_obs
        
        reward_list = []
        avg_reward_list = []
        #loop over each game and reset the game environment
        for game_iterator in range(self.epochs):
            state = self.env.reset()
            state, reward, is_game_done = self.collect_states(action=0)

            is_game_done = False
            tot_reward, reward = 0,0

            for frame_iterator in range(M_MAX_FRAMES_PER_GAME):
                # self.env.render()
                action = self.q_agent.perform_action(state)
                next_state, reward, is_game_done = self.collect_states(action=action)

                self.q_agent.store_in_memory(state, action, reward, next_state, is_game_done)
                state = next_state

                if len(self.q_agent.state_list) > self.replay_batch_size and frame_iterator % self.replay_modulo == 0:
                    if self.use_target_network:
                        self.q_agent.replay_from_memory_double(self.replay_batch_size, num_observations_state)
                        self.q_agent.transfer_weights()
                    else:
                        self.q_agent.replay_from_memory(self.replay_batch_size, num_observations_state)
                tot_reward += reward

                if is_game_done:
                    break

                
            self.q_agent.update_epsilon_greedy()
            print("Game Number: {} of {}, Exploration: {:.2}, Frames Used: {}, Total reward: {}".format(game_iterator, self.epochs, self.q_agent.epsilon_max, frame_iterator, tot_reward))
            
            reward_list.append(tot_reward)
            if (game_iterator+1) % R_AVG_RANGE == 0:
                avg_reward = np.mean(reward_list)
                avg_reward_list.append(avg_reward)
                reward_list = []
                self.plot_results(avg_reward_list)


        self.env.close()
        self.q_agent.model.save(N_MODEL_FILE_PATH + N_MODEL_FILE_NAME)