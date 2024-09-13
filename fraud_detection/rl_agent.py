import gym
from stable_baselines3 import PPO
import numpy as np

class RLAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=10000)

    def predict(self, obs):
        action, _states = self.model.predict(obs)
        return action
