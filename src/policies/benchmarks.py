"""Causal benchmark policies; no access to future prices or execution flags."""
from dataclasses import dataclass
from math import isfinite, isclose
import random

IDS = tuple(f'B{i}' for i in range(8))

@dataclass(frozen=True)
class BenchmarkConfig:
    stop_loss: float = .05
    take_profit: float = .10
    trailing_stop: float = .05
    interval: int = 15
    seed: int = 42
    def __post_init__(self):
        for x in (self.stop_loss, self.trailing_stop):
            if not isfinite(x) or not 0 < x < 1:
                raise ValueError('invalid loss threshold')
        if not isfinite(self.take_profit) or self.take_profit <= 0:
            raise ValueError('invalid profit threshold')
        if type(self.interval) is not int or self.interval < 1:
            raise ValueError('invalid interval')
        if type(self.seed) is not int:
            raise ValueError('invalid seed')

class BenchmarkPolicy:
    def __init__(self, strategy, config=None):
        if strategy not in IDS:
            raise ValueError('unknown benchmark')
        self.strategy = strategy
        self.config = config or BenchmarkConfig()
        self.initial = None

    def reset(self, initial_close, *, seed=None):
        self._validate_price(initial_close)
        self.initial = self.peak = float(initial_close)
        self.next_step = 0
        self.rng = random.Random(self.config.seed if seed is None else seed)

    @staticmethod
    def _validate_price(price):
        if not isfinite(price) or price <= 0:
            raise ValueError('invalid close')

    def act(self, *, close, step, horizon):
        if self.initial is None:
            raise RuntimeError('reset first')
        self._validate_price(close)
        if type(step) is not int or type(horizon) is not int or not 0 <= step < horizon or step != self.next_step:
            raise ValueError('invalid or nonsequential step')
        if step == 0 and close != self.initial:
            raise ValueError('initial close mismatch')
        self.next_step += 1
        self.peak = max(self.peak, close)
        c = self.config
        b = self.strategy
        def le(a, z): return a < z or isclose(a, z, rel_tol=1e-12)
        sl = le(close, self.initial * (1-c.stop_loss))
        tp = le(self.initial * (1+c.take_profit), close)
        if b == 'B0': return 0
        if b == 'B1': return 4
        if b == 'B2': return 4 if sl else 0
        if b == 'B3': return 4 if tp else 0
        if b == 'B4': return 4 if sl or tp else 0
        if b == 'B5': return 4 if le(close, self.peak*(1-c.trailing_stop)) else 0
        if b == 'B6':
            if step == horizon-1: return 4
            if (step+1) % c.interval: return 0
            return 1 if step+1 == c.interval else 2
        return self.rng.randrange(5)
