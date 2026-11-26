"""Run real-data 60-step episodes in both reward modes, no training or file writes."""
from pathlib import Path
import math
import pandas as pd
from src.envs.optimal_exit_env import OptimalExitEnv
from src.envs.rewards import RewardConfig
from src.envs.accounting import CostConfig

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT/'data/processed/episodes/episode_manifest.csv'
s1_path = ROOT/'data/processed/s1/005930_s1.parquet'
for p in (manifest_path,s1_path):
    if not p.exists():
        raise FileNotFoundError(f'Missing input: {p}')
m = pd.read_csv(manifest_path, dtype={'symbol':str})
m['symbol'] = m['symbol'].str.zfill(6)
match = m[(m['symbol']=='005930') & (m['split']=='test')]
if match.empty:
    raise ValueError('No Samsung test episode in manifest')
ep = match.iloc[0]
d = pd.read_parquet(s1_path)
start = d.index.get_loc(pd.Timestamp(ep['start_date']))
horizon = int(ep['decision_steps'])
window = d.iloc[start:start+horizon+1]
if len(window)!=horizon+1 or str(window.index[-1].date()) != str(ep['last_execution_date'])[:10]:
    raise AssertionError('Episode rows do not match manifest')
print('episode:',ep['episode_id'])
print('window:',str(window.index[0].date()),'to',str(window.index[-1].date()))

results=[]
for mode in ('R1','R2'):
    env = OptimalExitEnv(window,horizon=horizon,initial_shares=100,
                         costs=CostConfig(commission_rate=0.001,sell_tax_rate=0.002,slippage_rate=0.001),
                         reward_config=RewardConfig(mode=mode,drawdown_lambda=0.5))
    obs,meta=env.reset(seed=42)
    total=0.;penalties=0.;steps=0
    for i in range(horizon):
        action=1 if i==0 else 0  # sell 25% on first decision, then hold
        obs,reward,term,trunc,info=env.step(action)
        total+=float(reward)
        penalties+=float(info['risk_penalty'])
        steps+=1
        if i==0:
            assert info['shares_sold']==25
        if term or trunc:break
    assert steps==horizon and not term and trunc
    assert info['shares_remaining']==75 and not info['forced_sale']
    assert math.isclose(info['terminal_value'],info['cash']+75*float(window.iloc[-1]['Close']),rel_tol=1e-12)
    if mode=='R1':
        assert math.isclose(total,math.log(info['terminal_value']/meta['initial_value']),abs_tol=1e-10)
    results.append((mode,total,penalties,info['terminal_value'],info['shares_remaining']))
    print(f'{mode}: steps={steps}, reward_sum={total:.9f}, risk_penalty_sum={penalties:.9f}, terminal={info["terminal_value"]:.3f}, remaining={info["shares_remaining"]}')
assert math.isclose(results[0][3],results[1][3],rel_tol=1e-12)
assert results[1][1] <= results[0][1]+1e-12
print('PASS: R1/R2 integration, accounting parity, no forced liquidation')
