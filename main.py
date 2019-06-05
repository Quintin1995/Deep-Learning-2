import gym
from network import DENSENET
from agent import Agent
from parameters import *
import numpy as np


#run a q-learning experiment
def run_experiment(amount_actions, num_observations_state):
    #loop over each game and reset the game environment
    for game_iterator in range(M_NUM_GAMES):
        state = env.reset()
        state = np.reshape(state, [1, num_observations_state])
        for frame_iterator in range(M_MAX_FRAMES_PER_GAME):
            #renders the game if set to true
            if M_DO_RENDER_GAME:
                env.render()
            action = q_learning_agent.perform_action(state)
            next_state, reward, is_game_done, _ = env.step(action)

            #penalize for ending the game
            if not is_game_done:
                reward = reward
            else:
                reward = -10

            next_state = np.reshape(next_state, [1, num_observations_state])
            state      = np.reshape(state, [1, num_observations_state])

            q_learning_agent.store_in_memory(state, action, reward, next_state, is_game_done)
            state = next_state
            if is_game_done:
                print("Game Number: {} of {}, Reward: {}, exploration: {:.3}".format(game_iterator, M_NUM_GAMES, frame_iterator, q_learning_agent.epsilon_max))
                break
            if len(q_learning_agent.state_list) > M_PLAY_BATCH_SIZE:
                q_learning_agent.replay_from_memory(M_PLAY_BATCH_SIZE, num_observations_state)
        q_learning_agent.update_epsilon_greedy()
    env.close()


##########################


#create the environment of the game
env = gym.make("LunarLander-v2")

#OBSERVATIONS in the game are structured like this:
## [position-x, position-y, velocity-x, velocity-y, lander-angle, lander-angular-velocity, leg0-ground-contact(boolean), leg1-ground-contact(boolean)]

print("Welcome to the game Lunar Lander Version-2!!!")
print("Games to be played")

amount_actions = env.action_space.n
num_observations_state  = env.observation_space.shape[0]

#create a q-learning agent
q_learning_agent = Agent(amount_actions, num_observations_state)

run_experiment(amount_actions, num_observations_state)