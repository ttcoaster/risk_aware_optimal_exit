"""Economic metrics for sell-only, zero-interest, fixed-horizon episodes."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StepRecord:
    value: float
    shares: int
    cash: float
    action: int
    execution_allowed: bool
    cost: float


def episode_metrics(initial_value, initial_shares, records, horizon=60):
    """Records are consecutive post-execution close marks; cost includes slippage."""
    rows = list(records)
    if not isinstance(horizon, int) or not 0 < len(rows) <= horizon or initial_shares <= 0 or not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError('invalid episode')
    prev = initial_shares
    prev_cash = 0.
    for i, r in enumerate(rows):
        if not np.isfinite([r.value,r.cash,r.cost]).all() or r.value <= 0 or not 0 <= r.cash <= r.value + 1e-8 or r.cost < 0 or not isinstance(r.shares, (int,np.integer)) or not 0 <= r.shares <= prev or r.action not in range(5) or r.cash < prev_cash - 1e-8:
            raise ValueError('invalid record')
        expected = prev if r.action == 4 else prev * r.action // 4
        sold = prev-r.shares
        if sold != (expected if r.execution_allowed else 0) or (sold == 0 and (r.cost != 0 or not np.isclose(r.cash,prev_cash))) or (r.shares == 0 and not np.isclose(r.value,r.cash)) or (r.shares == 0 and i != len(rows)-1):
            raise ValueError('inconsistent execution')
        prev, prev_cash = r.shares, r.cash
    if len(rows)<horizon and prev != 0:
        raise ValueError('incomplete episode with open position')
    values=np.array([initial_value]+[r.value for r in rows]+[rows[-1].value]*(horizon-len(rows)))
    returns=values[1:]/values[:-1]-1
    dd=1-values/np.maximum.accumulate(values)
    weights=np.array([1.]+[max(0.,1-r.cash/r.value) for r in rows]+[0.]*(horizon-len(rows)))
    sold=np.diff([initial_shares]+[r.shares for r in rows])<0
    def milestone(f):
        return next((i for i,r in enumerate(rows,1) if 1-r.shares/initial_shares >= f-1e-12),None)
    sd=float(np.std(returns,ddof=1)) if horizon>1 else 0.
    downside=float(np.sqrt(np.mean(np.minimum(returns,0)**2)))
    requests=sum(r.action!=0 for r in rows)
    fills=int(sold.sum())
    return dict(terminal_return=float(values[-1]/values[0]-1), log_return=float(np.log(values[-1]/values[0])), terminal_value=float(values[-1]), mdd=float(dd.max()), daily_volatility=sd, downside_deviation=downside, daily_sharpe=float(returns.mean()/sd) if sd>1e-15 else None, daily_sortino=float(returns.mean()/downside) if downside>1e-15 else None, steps=len(rows), horizon=horizon, first_sale_step=next((i for i,x in enumerate(sold,1) if x),None), exit_50_step=milestone(.5), exit_90_step=milestone(.9), exit_100_step=milestone(1.), remaining_fraction=prev/initial_shares, mean_stock_weight=float(weights[:-1].mean()), fills=fills, sell_requests=requests, noop_rate=(requests-fills)/requests if requests else None, cancelled_orders=sum(r.action!=0 and not r.execution_allowed for r in rows), total_cost=float(sum(r.cost for r in rows)), cost_fraction=float(sum(r.cost for r in rows)/initial_value))


def distribution_metrics(terminal_returns, alpha=.05, risk_aversion=3.):
    """Equal-weight episode distribution; positive loss convention; mean-variance CEQ."""
    x=np.asarray(terminal_returns,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or (x<=-1).any() or not 0<alpha<=1 or not np.isfinite(risk_aversion) or risk_aversion<0:
        raise ValueError('invalid distribution')
    losses=np.sort(-x)[::-1]; mass=alpha*len(x); k=int(np.ceil(mass))
    tail=(losses[:k-1].sum()+(mass-k+1)*losses[k-1])/mass
    return dict(n=len(x),mean_return=float(x.mean()),variance=float(x.var()),var_loss=float(losses[k-1]),cvar_loss=float(tail),ceq=float(x.mean()-risk_aversion*x.var()/2),alpha=alpha,risk_aversion=risk_aversion)


def paired_excess(policy, benchmark):
    """Mappings keyed by episode ID; reject unmatched samples."""
    if not policy or policy.keys()!=benchmark.keys():
        raise ValueError('episode IDs must match')
    out={k:float(policy[k]-benchmark[k]) for k in policy}
    if not np.isfinite(list(out.values())).all(): raise ValueError('nonfinite excess')
    return out
