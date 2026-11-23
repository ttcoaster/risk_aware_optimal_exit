"""Reward accounting for E1 v0.5; drawdown measured on marked portfolio value."""
from dataclasses import dataclass
from math import isfinite, log

@dataclass(frozen=True)
class RewardConfig:
    mode: str = 'R1'
    drawdown_lambda: float = 0.5

    def __post_init__(self):
        if self.mode not in ('R1', 'R2'):
            raise ValueError('mode must be R1 or R2')
        if not isfinite(self.drawdown_lambda) or self.drawdown_lambda < 0:
            raise ValueError('drawdown_lambda must be finite and nonnegative')

@dataclass(frozen=True)
class RewardResult:
    reward: float
    log_return: float
    drawdown_before: float
    drawdown_after: float
    drawdown_increase: float
    penalty: float
    peak_value: float


def compute_reward(old_value: float, new_value: float, peak_value: float,
                   config: RewardConfig) -> RewardResult:
    if not all(isfinite(v) and v > 0 for v in (old_value, new_value, peak_value)):
        raise ValueError('portfolio and peak values must be positive and finite')
    if peak_value + 1e-8 * peak_value < old_value:
        raise ValueError('peak_value must be at least old_value')
    new_peak = max(peak_value, old_value, new_value)
    old_dd = max(0., 1. - old_value / peak_value)
    new_dd = max(0., 1. - new_value / new_peak)
    increase = max(0., new_dd - old_dd)
    lr = log(new_value / old_value)
    penalty = config.drawdown_lambda * increase if config.mode == 'R2' else 0.
    return RewardResult(lr - penalty, lr, old_dd, new_dd, increase, penalty, new_peak)
