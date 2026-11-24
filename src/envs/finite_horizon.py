import gymnasium as gym

class FiniteHorizon(gym.Wrapper):
    """Only for E1: underlying truncation denotes the economic terminal horizon."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            if info.get('terminal_value') is None:
                raise RuntimeError('unexpected truncation without terminal valuation')
            info = dict(info, horizon_reached=True)
            terminated, truncated = True, False
        return obs, reward, terminated, truncated, info
