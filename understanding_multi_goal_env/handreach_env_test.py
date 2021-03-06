import gym
import tensorflow as tf
import numpy as np

env = gym.make('HandReach-v0')
env.reset()
done = False
desire_goal = None
achieved_goal = None
while not done:
	obs, rew, done, info = env.step(env.action_space.sample())
	print(rew)
	print(">>>>>>>>>>>>>")
	env.render()
	desire_goal = obs['desired_goal']
	achived_goal = obs['achieved_goal']
	print(desire_goal)
	print(">................................")
	print(achived_goal)
	print(done)
	

