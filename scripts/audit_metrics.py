"""Read-only train-window integration smoke; no model training or test windows."""
from pathlib import Path
from src.evaluation.metrics import StepRecord, episode_metrics, distribution_metrics
from collections import defaultdict
from math import isclose, log
from src.data.episode_loader import EpisodeLoader
from src.envs.optimal_exit_env import OptimalExitEnv
from src.envs.accounting import CostConfig
from src.policies.benchmarks import IDS, BenchmarkPolicy


def main():
    root = Path(__file__).resolve().parents[1]
    loader = EpisodeLoader(root/'data/processed/episodes/episode_manifest.csv', root/'data/processed/s1')
    records = {}
    for r in loader.records('train'):
        records.setdefault(r.symbol, r)
    assert records
    costs = CostConfig(commission_rate=.001, sell_tax_rate=.002, slippage_rate=.001)
    count = 0
    results = defaultdict(list)
    for r in records.values():
        frame = loader.load(r.episode_id, expected_split='train')
        for b in IDS:
            env = OptimalExitEnv(frame, horizon=r.decision_steps, initial_shares=100, costs=costs)
            _, info = env.reset(seed=42)
            initial = info['initial_value']
            policy = BenchmarkPolicy(b)
            policy.reset(float(frame.iloc[0]['Close']))
            total = 0.; remaining = 100
            rows = []
            previous_cash = 0.
            for t in range(r.decision_steps):
                action = policy.act(close=float(frame.iloc[t]['Close']), step=t, horizon=r.decision_steps)
                _, reward, terminated, truncated, info = env.step(action)
                total += reward
                assert remaining-info['shares_sold'] == info['shares_remaining']
                assert not info['forced_sale']
                if not info['execution_allowed']: assert info['shares_sold'] == 0
                remaining = info['shares_remaining']
                # Opportunity cost relative to unadjusted execution-day open.
                gross = info['shares_sold'] * float(frame.iloc[t+1]['Open']) if info['shares_sold'] else 0.
                cost = max(0., gross - (info['cash'] - previous_cash))
                previous_cash = info['cash']
                rows.append(StepRecord(info['portfolio_value'], remaining, info['cash'], action, bool(info['execution_allowed']), cost))
                if terminated or truncated: break
            assert terminated or truncated
            terminal = info['terminal_value']
            assert isclose(total, log(terminal/initial), abs_tol=1e-10)
            if b == 'B0': assert remaining == 100 and t+1 == r.decision_steps
            if b == 'B1' and frame.iloc[1:]['Tradable'].any(): assert remaining == 0
            m = episode_metrics(initial, 100, rows, r.decision_steps)
            assert isclose(m['log_return'], total, abs_tol=1e-10)
            results[b].append(m['terminal_return'])
            print(f"{r.symbol} {b}: return={m['terminal_return']:.6f}, mdd={m['mdd']:.6f}, exposure={m['mean_stock_weight']:.6f}, cost={m['cost_fraction']:.6f}")
            count += 1
            env.close()
    for b, returns in results.items():
        print(b, distribution_metrics(returns))
    print(f'PASS: {count} metric runs, train only, fixed horizon')

if __name__ == '__main__': main()
