"""Split-isolated episodic environment with fixed normalization and drawdown state."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.data.episode_loader import EpisodeSampler
from src.envs.normalized_observation import NormalizedObservation

class TrainingEnv(gym.Env):
    def __init__(self, loader, scaler, *, split='train', seed=42,
                 initial_shares=100, costs=None, reward_config=None):
        super().__init__()
        self.loader, self.scaler, self.split = loader, scaler, split
        self.sampler = EpisodeSampler(loader, split, seed)
        self.initial_shares, self.costs, self.reward_config = initial_shares, costs, reward_config
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(np.array([-scaler.clip]*5+[0]*4,dtype=np.float32),
            np.array([scaler.clip]*5+[1]*4,dtype=np.float32), dtype=np.float32)
        self._env = None
        self._done = True
        self.episode_id = None

    def _augment(self, obs, dd):
        result = np.append(obs, np.float32(dd)).astype(np.float32)
        if not self.observation_space.contains(result):
            raise ValueError('invalid augmented observation')
        return result

    def _metadata(self, info):
        rec = self.loader.record(self.episode_id)
        return dict(info, episode_id=rec.episode_id, symbol=rec.symbol, split=rec.split)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sampler = EpisodeSampler(self.loader, self.split, seed)
        options = options or {}
        eid = options.get('episode_id')
        if eid is None:
            eid = self.sampler.sample_id()
        raw = self.loader.make_env(eid, expected_split=self.split, initial_shares=self.initial_shares,
                                   costs=self.costs, reward_config=self.reward_config)
        env = NormalizedObservation(raw, self.scaler)
        obs, info = env.reset(seed=seed)
        result = self._augment(obs, 0.)
        if self._env is not None:
            self._env.close()
        self._env, self.episode_id, self._done = env, eid, False
        return result, self._metadata(dict(info, drawdown=0.))

    def step(self, action):
        if self._done or self._env is None:
            raise RuntimeError('reset required before step')
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._done = terminated or truncated
        return self._augment(obs, info['drawdown']), reward, terminated, truncated, self._metadata(info)

    def close(self):
        if self._env is not None:
            self._env.close()
        self._env = None
        self._done = True
