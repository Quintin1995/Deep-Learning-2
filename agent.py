from collections import *
from network import DENSENET
import numpy as np
import random
from parameters import *


class Agent:
    def __init__(self, amount_actions, amount_states):
        #create and build the denseNet model for the agent
        self.model = DENSENET()
        self.model = self.model.build_model()

        #set amount actions and amount states for the agent.
        self.amount_actions = amount_actions
        self.amount_states  = amount_states

        #epsilon params - used for exploring the state space.
        self.epsilon_max    = Q_MAX_EPSILON
        self.epsilon_min    = Q_MIN_EPSILON
        self.epsilon_decay  = Q_DECAY_EPSILON

        #learning rate
        self.learning_rate  = Q_LEARNING_RATE

        #create empty training set for the agent.
        self.state_list = deque(maxlen=Q_MAX_STATES_RETAINED)

        #used in the q-learning formula as the discount rate
        self.gamma = Q_GAMMA


    #add a state to the list of games states that should be remembered
    def store_in_memory(self, state, action, reward, next_state, is_game_over):
        self.state_list.append((state, action, reward, next_state, is_game_over))


    #return an index of the action to be taken at a given state.
    def perform_action(self, state):
        prob = np.random.rand()
        if prob <= self.epsilon_max:
            return random.randrange(self.amount_actions)
        #action values are the values of each neuron in the last layer of the neural network.
        action_values = self.model.predict(state)
        idx_action = np.argmax(action_values[0]) 
        return idx_action


    #update the epsilon value used for greedy search
    def update_epsilon_greedy(self):
        if self.epsilon_max > self.epsilon_min:
            self.epsilon_max = self.epsilon_max * self.epsilon_decay


    # Trains the neural network on a batch of states from memory of the agent.
    # A random sample form the state list (READ state-pair list) of the agent
    #  is picked and for each state-pair the new valuation of the current step
    #  is calculated with the help of the valuation of the next state.
    def replay_from_memory(self, batch_size, num_observations_state):
        batch = random.sample(self.state_list, batch_size)
        #loop over the total batch 
        for state, action, reward, next_state, is_game_over in batch:
            #reward of winning the game is given by the game itself
            target = reward
            if not is_game_over:
                #if the game is not over we apply the q-learning formula - Add the reward to the discounted predicted valuation of the next state to become the target valuation of the current state.
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            #Determine what the valuation is of the current state.
            target_current_state = self.model.predict(state)
            #train the network of the current state on the new target of the current state
            target_current_state[0][action] = target
            self.model.fit(state, target_current_state, epochs=1, verbose=0)