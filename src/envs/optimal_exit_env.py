"""E1 v0.5: single-stock, sell-only Gymnasium environment.

Decision at close t -> execution at open t+1 -> mark-to-market at close t+1.
60 actions require 61 consecutive source rows. No terminal forced sale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.envs.accounting import CostConfig, SellOnlyAccount
from src.envs.rewards import RewardConfig, compute_reward

FEATURES = ("return_1d", "return_5d", "return_20d", "volatility_20d", "ma_gap_20d")
REQUIRED = set(FEATURES) | {"Open", "Close", "Tradable", "FeatureReady"}


class OptimalExitEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, frame: pd.DataFrame, *, horizon: int = 60,
        initial_shares: int = 100, costs: CostConfig | None = None,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__()
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("frame index must be sorted unique DatetimeIndex")
        if REQUIRED - set(frame.columns):
            raise ValueError(f"missing columns: {sorted(REQUIRED - set(frame.columns))}")
        if type(horizon) is not int or horizon < 1 or len(frame) != horizon + 1:
            raise ValueError("frame must have exactly horizon + 1 rows")
        if type(initial_shares) is not int or initial_shares <= 0:
            raise ValueError("initial_shares must be a positive integer")
        if not bool(frame.iloc[0]["FeatureReady"]) or not bool(frame.iloc[0]["Tradable"]):
            raise ValueError("first decision row must be tradable and feature-ready")
        if frame[["Close"]].isna().any().any() or not np.isfinite(frame["Close"].to_numpy(dtype=float)).all() or (frame["Close"] <= 0).any():
            raise ValueError("all close marks must be positive and finite")
        if frame["Tradable"].isna().any() or frame["FeatureReady"].isna().any():
            raise ValueError("invalid trading flags")
        if not frame["Tradable"].isin([True, False]).all() or not frame["FeatureReady"].isin([True, False]).all():
            raise ValueError("trading flags must be boolean")
        if (frame["FeatureReady"] & ~frame["Tradable"]).any():
            raise ValueError("feature-ready row cannot be nontradable")
        if frame.loc[frame["FeatureReady"], list(FEATURES)].isna().any().any():
            raise ValueError("missing feature on ready row")
        if not np.isfinite(frame.loc[frame["FeatureReady"], list(FEATURES)].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite feature on ready row")
        self.frame = frame.copy()
        self.horizon = horizon
        self.initial_shares = initial_shares
        self.costs = costs if costs is not None else CostConfig()
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self._peak_value = 0.0
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.account: SellOnlyAccount | None = None
        self._step_idx = 0
        self._done = False
        self._last_features: np.ndarray | None = None
        self._initial_value = 0.0

    def _observation(self) -> np.ndarray:
        row = self.frame.iloc[self._step_idx]
        if bool(row["FeatureReady"]):
            self._last_features = row[list(FEATURES)].to_numpy(dtype=np.float32)
        # If suspended, retain last known feature vector; no future information.
        assert self._last_features is not None and self.account is not None
        price = float(row["Close"])
        state = np.concatenate((self._last_features, np.array([
            self.account.stock_weight(price), self.account.remaining_fraction,
            min(self._step_idx / self.horizon, 1.0),
        ], dtype=np.float32)))
        return state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.account = SellOnlyAccount(self.initial_shares)
        self._step_idx = 0
        self._done = False
        self._last_features = None
        self._initial_value = self.account.market_value(float(self.frame.iloc[0]["Close"]))
        self._peak_value = self._initial_value
        return self._observation(), {"initial_value": self._initial_value}

    def step(self, action):
        if self.account is None or self._done:
            raise RuntimeError("call reset() before step(), and do not step after done")
        if not self.action_space.contains(action):
            raise ValueError("action must be 0..4")
        action = int(action)
        old_row = self.frame.iloc[self._step_idx]
        old_value = self.account.market_value(float(old_row["Close"]))
        next_row = self.frame.iloc[self._step_idx + 1]
        execution_allowed = bool(next_row["Tradable"]) and np.isfinite(float(next_row["Open"])) and float(next_row["Open"]) > 0
        sale = None
        if execution_allowed:
            sale = self.account.sell(action, float(next_row["Open"]), self.costs)
        # Unfilled orders are cancelled; no deferred execution.
        self._step_idx += 1
        new_value = self.account.market_value(float(next_row["Close"]))
        reward_result = compute_reward(old_value, new_value, self._peak_value, self.reward_config)
        self._peak_value = reward_result.peak_value
        reward = reward_result.reward
        terminated = self.account.shares == 0
        truncated = self._step_idx >= self.horizon and not terminated
        self._done = terminated or truncated
        observation = self._observation()
        info = {
            "date": str(self.frame.index[self._step_idx].date()),
            "execution_allowed": execution_allowed,
            "shares_sold": sale.shares_sold if sale else 0,
            "shares_remaining": self.account.shares,
            "cash": self.account.cash,
            "portfolio_value": new_value,
            "terminal_value": self.account.terminal_value(float(next_row["Close"])) if self._done else None,
            "forced_sale": False,
            "reward_mode": self.reward_config.mode,
            "log_return": reward_result.log_return,
            "drawdown": reward_result.drawdown_after,
            "drawdown_increase": reward_result.drawdown_increase,
            "risk_penalty": reward_result.penalty,
            "peak_value": self._peak_value,
        }
        return observation, reward, terminated, truncated, info
