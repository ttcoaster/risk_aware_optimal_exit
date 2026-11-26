"""Train-only paired integration audit; existing checkpoints, no learning."""
import argparse,csv,json,hashlib
from pathlib import Path
import stable_baselines3 as sb3
from src.data.episode_loader import EpisodeLoader
from src.data.observation_scaler import FeatureScaler
from src.envs.training_env import TrainingEnv
from src.envs.finite_horizon import FiniteHorizon
from src.envs.accounting import CostConfig
from src.envs.rewards import RewardConfig
from src.policies.benchmarks import IDS,BenchmarkPolicy
from src.evaluation.engine import evaluate_episode,episode_seed,summarize
ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-run',required=True); p.add_argument('--run-id',required=True); a=p.parse_args()
    if not a.model_run.isalnum() or not a.run_id.isalnum(): p.error('alphanumeric IDs required')
    model_dir=ROOT/'artifacts/agent_smoke'/a.model_run
    manifest=ROOT/'data/processed/episodes/episode_manifest.csv'; stats=ROOT/'artifacts/normalization/s1_train_scaler.json'
    inputs={str(x.relative_to(ROOT)):digest(x) for x in (manifest,stats)}
    if json.loads((model_dir/'inputs.json').read_text())!=inputs: raise RuntimeError('training input hashes differ')
    loader=EpisodeLoader(manifest,ROOT/'data/processed/s1'); scaler=FeatureScaler.load(stats)
    costs=CostConfig(commission_rate=.001,sell_tax_rate=.002,slippage_rate=.001)
    models={}
    for alg in ('PPO','A2C','DQN'):
        for mode in ('R1','R2'):
            name=f'{alg}_{mode}'; folder=model_dir/name; path=folder/'model.zip'
            meta=json.loads((folder/'summary.json').read_text())
            if meta['algorithm']!=alg or meta['reward']!=mode or meta['costs']!=vars(costs) or meta['drawdown_lambda']!=.5 or meta['status']!='PASS': raise RuntimeError('checkpoint metadata mismatch')
            models[name]=getattr(sb3,alg).load(path,device='cpu'); inputs[str(path.relative_to(ROOT))]=digest(path)
    selected={}
    for r in loader.records('train'): selected.setdefault(r.symbol,r)
    if not selected: raise RuntimeError('empty train split')
    rows=[]; traces=[]
    for r in selected.values():
        frame=loader.load(r.episode_id,expected_split='train'); seed=episode_seed(r.episode_id)
        for name in (*IDS,*models):
            mode=name.split('_')[-1] if name in models else 'R1'
            env=FiniteHorizon(TrainingEnv(loader,scaler,split='train',costs=costs,reward_config=RewardConfig(mode=mode,drawdown_lambda=.5)))
            try:
                if name in models:
                    model=models[name]
                    if model.observation_space!=env.observation_space or model.action_space!=env.action_space: raise RuntimeError('space mismatch')
                    def policy(obs,close,t,h): return model.predict(obs,deterministic=True)[0].item()
                else:
                    b=BenchmarkPolicy(name); b.reset(float(frame.iloc[0]['Close']),seed=seed)
                    def policy(obs,close,t,h): return b.act(close=close,step=t,horizon=h)
                m,trace=evaluate_episode(env,r,frame,name,policy,seed); rows.append(m); traces.extend(trace)
                print(f"{r.symbol} {name}: return={m['terminal_return']:.6f}, mdd={m['mdd']:.6f}, steps={m['steps']}")
            finally: env.close()
    summaries,pairs=summarize(rows)
    out=ROOT/'artifacts/evaluation'/a.run_id; out.mkdir(parents=True,exist_ok=False)
    for name,data in [('episodes',rows),('steps',traces),('summary',summaries),('paired_b0',pairs)]:
        with (out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2))
    print(f'PASS: {len(rows)} paired policy runs, train only; output={out}')
if __name__=='__main__': main()
