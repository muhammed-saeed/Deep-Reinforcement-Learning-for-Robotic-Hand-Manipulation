import gym
import tensorflow as tf
import numpy as np

env = gym.make('HandReach-v0')
env.reset()
done = False
while not done:
	obs, rew, done, info = env.step(env.action_space.sample())
	print(rew)
	print(">>>>>>>>>>>>>")
	env.render()

