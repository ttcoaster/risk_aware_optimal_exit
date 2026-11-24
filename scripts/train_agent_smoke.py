"""Train-only pilot; fixed validation inference, no model selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from src.data.episode_loader import EpisodeLoader
from src.data.observation_scaler import FeatureScaler
from src.envs.training_env import TrainingEnv
from src.envs.finite_horizon import FiniteHorizon
from src.envs.rewards import RewardConfig
from src.envs.accounting import CostConfig

ROOT = Path(__file__).resolve().parents[1]

def build_agent(name, env, seed):
    common = dict(policy='MlpPolicy', env=env, seed=seed, device='cpu', gamma=1.0,
                  learning_rate=3e-4, verbose=0, policy_kwargs=dict(net_arch=[64,64]))
    if name == 'PPO':
        return sb3.PPO(**common, n_steps=128, batch_size=64, n_epochs=4, gae_lambda=.95)
    if name == 'A2C':
        return sb3.A2C(**common, n_steps=16, gae_lambda=.95)
    if name == 'DQN':
        return sb3.DQN(**common, buffer_size=10000, learning_starts=128, batch_size=64,
                       train_freq=4, gradient_steps=1, target_update_interval=250,
                       exploration_fraction=.5)
    raise ValueError(name)

def run(name, mode, steps, seed, loader, scaler, out, val_count):
    out.mkdir(parents=True, exist_ok=False)
    costs = CostConfig(commission_rate=.001, sell_tax_rate=.002, slippage_rate=.001)
    reward_config = RewardConfig(mode=mode, drawdown_lambda=.5)
    def make(split):
        return FiniteHorizon(TrainingEnv(loader, scaler, split=split, seed=seed,
                                        costs=costs, reward_config=reward_config))
    train = Monitor(make('train'), str(out/'train.monitor.csv'))
    val = make('validation')
    try:
        model = build_agent(name, train, seed)
        model.set_logger(configure(str(out), ['csv']))
        before = {k:v.detach().clone() for k,v in model.policy.named_parameters()}
        model.learn(total_timesteps=steps, log_interval=1)
        delta = max(float((v.detach()-before[k]).abs().max()) for k,v in model.policy.named_parameters())
        if not all(torch.isfinite(v).all().item() for v in model.policy.parameters()) or not np.isfinite(delta) or delta <= 0:
            raise RuntimeError('nonfinite or unchanged parameters')
        model.save(out/'model')
        loaded = type(model).load(out/'model.zip', device='cpu')
        rows = []
        for rec in loader.records('validation')[:val_count]:
            obs, info = val.reset(seed=seed, options={'episode_id':rec.episode_id})
            initial = info['initial_value']; total=0.; penalty=0.; mdd=0.; actions=[]
            for _ in range(rec.decision_steps):
                a,_ = loaded.predict(obs, deterministic=True)
                original,_ = model.predict(obs, deterministic=True)
                if int(a) != int(original): raise RuntimeError('reload prediction mismatch')
                action=int(a); actions.append(action)
                obs,r,term,trunc,info=val.step(action)
                if info['split']!='validation' or info['episode_id']!=rec.episode_id or info['forced_sale'] or not np.isfinite(r) or not val.observation_space.contains(obs):
                    raise RuntimeError('invalid validation transition')
                total+=r; penalty+=info['risk_penalty']; mdd=max(mdd,info['drawdown'])
                if term or trunc: break
            if not term or trunc: raise RuntimeError('invalid terminal flags')
            terminal=info['terminal_value']; logret=float(np.log(terminal/initial))
            if not np.isclose(total,logret-penalty,atol=1e-8): raise RuntimeError('reward accounting mismatch')
            rows.append(dict(episode_id=rec.episode_id,symbol=rec.symbol,steps=len(actions),reward_sum=total,
                             terminal_return=terminal/initial-1,max_drawdown=mdd,risk_penalty=penalty,
                             remaining_shares=info['shares_remaining'],actions=json.dumps(actions)))
        if not rows: raise RuntimeError('empty validation')
        with (out/'validation.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        result=dict(algorithm=name,reward=mode,seed=seed,requested_steps=steps,actual_steps=model.num_timesteps,
                    max_parameter_change=delta,validation_episodes=len(rows),gamma=1.,drawdown_lambda=.5,
                    costs=vars(costs),sb3=sb3.__version__,torch=str(torch.__version__),status='PASS')
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
        return result
    finally:
        train.close(); val.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--algorithm',choices=['PPO','A2C','DQN','all'],default='all')
    p.add_argument('--reward',choices=['R1','R2','both'],default='both'); p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--validation-count',type=int,default=10)
    p.add_argument('--run-id',required=True); a=p.parse_args()
    if a.steps<256 or a.validation_count<1 or not a.run_id.isalnum(): p.error('steps >=256, validation-count >=1, alphanumeric run-id required')
    torch.set_num_threads(1)
    manifest=ROOT/'data/processed/episodes/episode_manifest.csv'; stats=ROOT/'artifacts/normalization/s1_train_scaler.json'
    loader=EpisodeLoader(manifest,ROOT/'data/processed/s1'); scaler=FeatureScaler.load(stats)
    out=ROOT/'artifacts/agent_smoke'/a.run_id; out.mkdir(parents=True,exist_ok=False)
    (out/'inputs.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest,stats]},indent=2))
    results=[]
    for name in (['PPO','A2C','DQN'] if a.algorithm=='all' else [a.algorithm]):
        for mode in (['R1','R2'] if a.reward=='both' else [a.reward]):
            results.append(run(name,mode,a.steps,a.seed,loader,scaler,out/f'{name}_{mode}',a.validation_count))
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print(f'PASS: {len(results)} agent runs, saved/reloaded, validation inference; no test episodes')
if __name__=='__main__': main()
