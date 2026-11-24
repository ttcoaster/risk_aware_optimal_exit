"""Gymnasium observation-only wrapper: reward, actions and accounting unchanged."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.data.observation_scaler import FeatureScaler

class NormalizedObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, scaler: FeatureScaler):
        super().__init__(env)
        self.scaler = scaler
        if env.observation_space.shape != (8,):
            raise ValueError('expected 8-dimensional observation')
        self.observation_space = spaces.Box(
            low=np.array([-scaler.clip]*5+[0.0]*3,dtype=np.float32),
            high=np.array([scaler.clip]*5+[1.0]*3,dtype=np.float32),
            dtype=np.float32)

    def observation(self, observation):
        result = self.scaler.transform(observation)
        if not self.observation_space.contains(result):
            raise ValueError('normalized observation outside declared bounds')
        return result
