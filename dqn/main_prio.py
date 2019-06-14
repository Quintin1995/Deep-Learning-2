import gym
from network import DENSENET
from agent import Agent
from parameters import *
import numpy as np
import os

# Format_State
def f_s(state, n) :
    return np.reshape(state, [1, n])

tn = "" # Target Network
qn = "" # Q-network

#run a q-learning experiment
def run_experiment(amount_actions, n):
    #loop over each game and reset the game environment
    for game_iterator in range(M_NUM_GAMES):
        state = env.reset()
        state = f_s(state, n)

        for frame_iterator in range(M_MAX_FRAMES_PER_GAME):
            # Show next frame 
            env.render()

            # Perform action at step
            action = q_learning_agent.perform_action(state)
            
            # Obtain state information from step
            next_state, reward, is_game_done, _ = env.step(action)


            q_learning_agent.store_in_memory(f_s(state, n), action, reward, f_s(next_state, n), is_game_done)
            
            state = next_state
            if is_game_done:
                break

            if len(q_learning_agent.state_list) > M_PLAY_BATCH_SIZE:
                q_learning_agent.replay_from_memory(M_PLAY_BATCH_SIZE, n)
                
        print("Game Number: {} of {}, Reward: {}, exploration: {:.3}".format(game_iterator, M_NUM_GAMES, frame_iterator, q_learning_agent.epsilon_max))
        q_learning_agent.update_epsilon_greedy()
    env.close()
    q_learning_agent.model.save(N_MODEL_FILE_PATH + N_MODEL_FILE_NAME)


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