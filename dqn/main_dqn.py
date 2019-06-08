import gym
from dqn.network import DENSENET
from dqn.agent import Agent
from dqn.parameters import *
import numpy as np
import os
import matplotlib.pyplot as plt

# Plots the average reward
def plot_results(avg_reward_list, name=R_PLOTS_FILE):
        # Plot Rewards
    plt.plot(10*(np.arange(len(avg_reward_list)) + 1), avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig(R_PLOTS_PATH+name)     
    plt.close()

#run a q-learning experiment
def run_experiment(amount_actions, num_observations_state):
    reward_list = []
    avg_reward_list = []

    #loop over each game and reset the game environment
    for game_iterator in range(M_NUM_GAMES+1):
        is_game_done = False
        tot_reward, reward = 0,0
        state = env.reset()
        state = np.reshape(state, [1, num_observations_state])
        for frame_iterator in range(M_MAX_FRAMES_PER_GAME):
            #renders the game if set to true
            if M_DO_RENDER_GAME and not (game_iterator % M_RENDER_GAME_MODULO):
                env.render()
            action = q_learning_agent.perform_action(state)
            next_state, reward, is_game_done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, num_observations_state])
            state      = np.reshape(state, [1, num_observations_state])

            q_learning_agent.store_in_memory(state, action, reward, next_state, is_game_done)
            state = next_state
            if is_game_done:
                break
            if len(q_learning_agent.state_list) > M_PLAY_BATCH_SIZE:
                q_learning_agent.replay_from_memory(M_PLAY_BATCH_SIZE, num_observations_state)
            tot_reward += reward
        print("Game Number: {} of {}, Exploration: {:.2}, Frames Used: {}, Reward: {:.4}".format(game_iterator, M_NUM_GAMES, q_learning_agent.epsilon_max, frame_iterator, tot_reward))
        q_learning_agent.update_epsilon_greedy()
        reward_list.append(tot_reward)
        if (game_iterator+1) % R_AVG_RANGE == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []
            plot_results(avg_reward_list)
    env.close()
    q_learning_agent.model.save(N_MODEL_FILE_PATH + N_MODEL_FILE_NAME)
    



if __name__ == '__main__':
    ##########################


    #create the environment of the game
    env = gym.make("LunarLander-v2")

    #OBSERVATIONS in the game are structured like this:
    ## [position-x, position-y, velocity-x, velocity-y, lander-angle, lander-angular-velocity, leg0-ground-contact(boolean), leg1-ground-contact(boolean)]

    print("Welcome to the game Lunar Lander Version-2!!!")
    print("Games to be played")

    amount_actions = env.action_space.n
    num_observations_state  = env.observation_space.shape[0]

    #create model save path if does not exist yet
    if not os.path.exists(N_MODEL_FILE_PATH):
        os.mkdir(N_MODEL_FILE_PATH)

    #create a q-learning agent
    q_learning_agent = Agent(amount_actions, num_observations_state)

    run_experiment(amount_actions, num_observations_state)
