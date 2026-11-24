"""Read-only train-window integration smoke; no model training or test windows."""
from pathlib import Path
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
    for r in records.values():
        frame = loader.load(r.episode_id, expected_split='train')
        for b in IDS:
            env = OptimalExitEnv(frame, horizon=r.decision_steps, initial_shares=100, costs=costs)
            _, info = env.reset(seed=42)
            initial = info['initial_value']
            policy = BenchmarkPolicy(b)
            policy.reset(float(frame.iloc[0]['Close']))
            total = 0.; remaining = 100
            for t in range(r.decision_steps):
                action = policy.act(close=float(frame.iloc[t]['Close']), step=t, horizon=r.decision_steps)
                _, reward, terminated, truncated, info = env.step(action)
                total += reward
                assert remaining-info['shares_sold'] == info['shares_remaining']
                assert not info['forced_sale']
                if not info['execution_allowed']: assert info['shares_sold'] == 0
                remaining = info['shares_remaining']
                if terminated or truncated: break
            assert terminated or truncated
            terminal = info['terminal_value']
            assert isclose(total, log(terminal/initial), abs_tol=1e-10)
            if b == 'B0': assert remaining == 100 and t+1 == r.decision_steps
            if b == 'B1' and frame.iloc[1:]['Tradable'].any(): assert remaining == 0
            print(f'{r.symbol} {b}: steps={t+1}, remaining={remaining}, return={terminal/initial-1:.6f}')
            count += 1
            env.close()
    print(f'PASS: {count} benchmark runs, train only, shared execution and accounting')

if __name__ == '__main__': main()
