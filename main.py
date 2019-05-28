import gym
import universe


env = gym.make("LunarLander-v2")

for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(len(observation))
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()