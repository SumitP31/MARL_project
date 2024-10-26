import gym
from stable_baselines3 import PPO
from environment import PressurePlate
import numpy as np

# Initialize the environment
env = PressurePlate(15,9,4,4,'linear')
obs = env.reset()

# action = [3,0,0,0]

# obs_, reward, terminated, info = env.step(action)
# action = [1,0,0,0]
# obs_, reward, terminated, info = env.step(action)

# # obs_ = np.reshape(obs_,(3,3))
# print(obs_[0])
# print("----------------\n----------------")
# print(reward[0])

env.render()



# Use PPO which supports tuple action spaces
# model = PPO('MlpPolicy', env, verbose=1)

# # Train the agent
# model.learn(total_timesteps=10000)

# # Save the model
# model.save("ppo_pressureplate_4p")

# # Test the agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()

# env.close()
