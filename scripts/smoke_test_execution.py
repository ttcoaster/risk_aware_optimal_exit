"""Real-data execution smoke test; run: python -m scripts.smoke_test_execution"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.envs.accounting import CostConfig
from src.envs.optimal_exit_env import OptimalExitEnv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/processed/episodes/episode_manifest.csv'
FEATURES = ROOT / 'data/processed/s1/005930_s1.parquet'


def main():
    manifest = pd.read_csv(MANIFEST, dtype={'symbol': str})
    row = manifest.loc[(manifest['symbol'] == '005930') & (manifest['split'] == 'test')].iloc[0]
    df = pd.read_parquet(FEATURES)
    window = df.loc[pd.Timestamp(row['start_date']):pd.Timestamp(row['last_execution_date'])]
    horizon = int(row['decision_steps'])
    assert len(window) == horizon + 1, f'Expected {horizon + 1} rows, got {len(window)}'
    costs = CostConfig(commission_rate=0.001, sell_tax_rate=0.002, slippage_rate=0.001)
    env = OptimalExitEnv(window, horizon=horizon, initial_shares=100, costs=costs)
    _, initial = env.reset(seed=42)
    # First day: sell 25% of the initial remaining holdings. Afterwards Hold.
    rewards = []
    for step in range(horizon):
        action = 1 if step == 0 else 0
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if step == 0:
            first = dict(info)
        if terminated or truncated:
            break
    assert first['shares_sold'] == 25
    assert info['shares_remaining'] == 75
    assert info['forced_sale'] is False
    assert truncated and not terminated
    assert np.isclose(sum(rewards), np.log(info['terminal_value']/initial['initial_value']))
    print('episode:', row['episode_id'])
    print('first execution date:', first['date'])
    print('first shares sold:', first['shares_sold'])
    print('first cash after costs:', round(first['cash'], 4))
    print('steps:', len(rewards))
    print('remaining shares:', info['shares_remaining'])
    print('terminal mark-to-market:', round(info['terminal_value'], 4))
    print('forced_sale:', info['forced_sale'])
    print('PASS: next-open partial sale, fees, no forced terminal liquidation')


if __name__ == '__main__':
    main()
