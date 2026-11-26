"""STEP 3-2 real-data Hold episode smoke test. Run from project root."""
from pathlib import Path
import pandas as pd
from src.envs.optimal_exit_env import OptimalExitEnv


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / 'data/processed/episodes/episode_manifest.csv'
    feature_path = root / 'data/processed/s1/005930_s1.parquet'
    for path in (manifest_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')

    manifest = pd.read_csv(manifest_path, dtype={'symbol': str})
    candidates = manifest.loc[(manifest['symbol'] == '005930') & (manifest['split'] == 'test')]
    if candidates.empty:
        raise ValueError('No 005930 test episode found in manifest')
    episode = candidates.iloc[0]
    data = pd.read_parquet(feature_path)
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    start = data.index.get_loc(pd.Timestamp(episode['start_date']))
    horizon = int(episode['decision_steps'])
    window = data.iloc[start:start + horizon + 1]
    if len(window) != horizon + 1:
        raise ValueError('Episode window shorter than horizon + 1')
    if pd.Timestamp(window.index[-1]) != pd.Timestamp(episode['last_execution_date']):
        raise ValueError('Episode end date does not match manifest')

    env = OptimalExitEnv(window, horizon=horizon, initial_shares=100)
    obs, info = env.reset(seed=42)
    print('episode:', episode['episode_id'])
    print('observation:', obs.shape, 'initial value:', info['initial_value'])
    for step in range(horizon):
        obs, reward, terminated, truncated, info = env.step(0)  # Hold
        if terminated or truncated:
            break
    print('steps:', step + 1)
    print('done:', terminated, truncated)
    print('remaining:', info['shares_remaining'])
    print('terminal:', info['terminal_value'])
    print('forced_sale:', info['forced_sale'])
    assert step + 1 == horizon
    assert not terminated and truncated
    assert info['shares_remaining'] == 100
    assert info['forced_sale'] is False
    print('PASS: 60-step Hold episode; no forced liquidation')


if __name__ == '__main__':
    main()
