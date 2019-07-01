from collections import *
from dqn.network import DENSENET, QNetwork
import numpy as np
import random
from dqn.parameters import *

class Agent:
    def __init__(self, amount_actions, amount_states, mdl_type, env, use_target=False, memory=10000):
        #create and build the denseNet model for the agent
        self.model = QNetwork(env,model_type=mdl_type)
        self.model = self.model.model
        if use_target: 
            self.target_model = QNetwork(env,model_type=mdl_type)
            self.target_model = self.target_model.model

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
        self.state_list = deque(maxlen=memory)

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

    def transfer_weights (self):
        W = self.model.get_weights()
        target_W = self.target_model.get_weights()
        for i in range (len (W)):
            target_W[i] = P_TAU * W[i] + (1 - P_TAU) * target_W[i]
        self.target_model.set_weights(target_W)

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
        states_batch = np.zeros((batch_size, 84, 84, 4))
        target_batch = np.zeros((batch_size, 4))

        idx = 0

        for state, action, reward, next_state, is_game_over in batch:
            #reward of winning the game is given by the game itself
            target = self.model.predict(state)
            if is_game_over:
                target[0][action] = reward
            else:
                #if the game is not over we apply the q-learning formula - Add the reward to the discounted predicted valuation of the next state to become the target valuation of the current state.
                prediction = self.model.predict(next_state)
                Q_future = max(prediction[0])
                target[0][action] = (reward + self.gamma * Q_future)
                
            states_batch[idx] = state
            target_batch[idx] = target
            idx += 1

        self.model.fit(states_batch, target_batch, epochs=1, verbose=0) 

    # Trains the neural network on a batch of states from memory of the agent.
    # A random sample form the state list (READ state-pair list) of the agent
    #  is picked and for each state-pair the new valuation of the current step
    #  is calculated with the help of the valuation of the next state.
    def replay_from_memory_double(self, batch_size, num_observations_state):
        batch = random.sample(self.state_list, batch_size)
        #loop over the total batch 
        states_batch = np.zeros((batch_size, num_observations_state))
        target_batch = np.zeros((batch_size, 4))

        idx = 0

        for state, action, reward, next_state, is_game_over in batch:
            #reward of winning the game is given by the game itself
            q_val   = self.model.predict(state)
            q_target= self.target_model.predict(next_state)[0]

            if is_game_over:
                q_val[0][action] = reward
            else:
                #if the game is not over we apply the q-learning formula - Add the reward to the discounted predicted valuation of the next state to become the target valuation of the current state.
                q_next = self.model.predict(next_state)
                q_next = q_next[0]
                next_best_action = np.argmax(q_next)
                q_val[0][action] = (reward + self.gamma * q_target[next_best_action])
                
            states_batch[idx] = state
            target_batch[idx] = q_val
            idx += 1

        self.model.fit(states_batch, target_batch, epochs=1, verbose=0) 