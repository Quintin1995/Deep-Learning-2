import gym
import universe
from network import DENSENET
from parameters import *


#before we start the game we create an empty new model of the class DENSENET, described in network.py
model = DENSENET()
model = model.build_model()


env = gym.make("LunarLander-v2")
for i_episode in range(M_NUM_GAMES):
    observation = env.reset()
    for t in range(M_MAX_FRAMES_PER_GAME):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


