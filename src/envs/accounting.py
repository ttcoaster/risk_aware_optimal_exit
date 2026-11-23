"""Deterministic, sell-only accounting for E1 Optimal Exit.

This module has no market-data or Gymnasium dependency.
Cash and prices use currency units; quantities are whole shares.
"""
from dataclasses import dataclass
from math import floor, isfinite

ACTION_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.0
    sell_tax_rate: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self):
        for name in ("commission_rate", "sell_tax_rate", "slippage_rate"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")


@dataclass(frozen=True)
class SaleResult:
    action: int
    shares_sold: int
    execution_price: float
    gross_proceeds: float
    commission: float
    sell_tax: float
    net_proceeds: float


class SellOnlyAccount:
    def __init__(self, initial_shares: int, initial_cash: float = 0.0):
        if type(initial_shares) is not int or initial_shares <= 0:
            raise ValueError("initial_shares must be a positive integer")
        if not isfinite(initial_cash) or initial_cash < 0:
            raise ValueError("initial_cash must be finite and nonnegative")
        self.initial_shares = initial_shares
        self.shares = initial_shares
        self.cash = float(initial_cash)

    def market_value(self, mark_price: float) -> float:
        if not isfinite(mark_price) or mark_price <= 0:
            raise ValueError("mark_price must be positive and finite")
        return self.cash + self.shares * mark_price

    @property
    def remaining_fraction(self) -> float:
        return self.shares / self.initial_shares

    def stock_weight(self, mark_price: float) -> float:
        value = self.market_value(mark_price)
        return self.shares * mark_price / value

    def sell(self, action: int, open_price: float, costs: CostConfig) -> SaleResult:
        if type(action) is not int or not 0 <= action < len(ACTION_FRACTIONS):
            raise ValueError("action must be an integer from 0 to 4")
        if not isfinite(open_price) or open_price <= 0:
            raise ValueError("open_price must be positive and finite")
        if not isinstance(costs, CostConfig):
            raise TypeError("costs must be CostConfig")
        fraction = ACTION_FRACTIONS[action]
        quantity = self.shares if action == 4 else floor(self.shares * fraction)
        execution_price = open_price * (1 - costs.slippage_rate)
        gross = quantity * execution_price
        commission = gross * costs.commission_rate
        sell_tax = gross * costs.sell_tax_rate
        net = gross - commission - sell_tax
        self.shares -= quantity
        self.cash += net
        return SaleResult(action, quantity, execution_price, gross, commission, sell_tax, net)

    def terminal_value(self, mark_price: float) -> float:
        """Mark to market without selling remaining shares or charging exit fees."""
        return self.market_value(mark_price)
