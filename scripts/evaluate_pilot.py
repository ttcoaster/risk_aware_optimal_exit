"""Evaluate every pilot checkpoint and fresh benchmarks on validation only."""
import argparse,json
from pathlib import Path
from src.experiments.pilot import read,checkpoints
from src.experiments.training_engine import verify_plan,write_json
from src.experiments.configuration import require,digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--training-run',required=True); p.add_argument('--run-id',required=True); a=p.parse_args()
    if not all(x.isascii() and x.isalnum() for x in vars(a).values()): p.error('ASCII alphanumeric IDs required')
    import pandas as pd
    import torch
    import stable_baselines3 as sb3
    from src.experiments.checkpoint_tracking import parameter_hash
    from src.data.episode_loader import EpisodeLoader
    from src.data.observation_scaler import FeatureScaler
    from src.envs.training_env import TrainingEnv
    from src.envs.finite_horizon import FiniteHorizon
    from src.envs.accounting import CostConfig
    from src.envs.rewards import RewardConfig
    from src.policies.benchmarks import IDS,BenchmarkPolicy
    from src.evaluation.engine import evaluate_episode,episode_seed,summarize
    root=Path(__file__).resolve().parents[1]; folder=root/'artifacts/tracked_training'/a.training_run
    plan=read(folder/'plan.json'); inputs=verify_plan(root,plan); registry=checkpoints(plan,folder)
    out=root/'artifacts/evaluation'/a.run_id; tmp=out.with_name(out.name+'.partial')
    require(not out.exists(),'output exists'); tmp.mkdir(parents=True,exist_ok=False)
    cfg=plan['config']; e=cfg['environment']; torch.set_num_threads(cfg['experiment']['torch_threads'])
    loader=EpisodeLoader(root/e['manifest'],root/e['feature_dir']); scaler=FeatureScaler.load(root/e['scaler']); recs=loader.records('validation')
    require(len(recs)==376 and all(r.decision_steps==60 for r in recs),'validation grid')
    specs=[(b,'R1',None,None) for b in IDS]; hashes={}
    for j,x,path in registry:
        model=getattr(sb3,j['algorithm']).load(path,device='cpu')
        require(model.num_timesteps==x['step'] and parameter_hash(model.policy)==x['parameter_sha256'],'loaded checkpoint mismatch')
        name=f"{j['algorithm']}_{j['reward']}_{x['step']}"; specs.append((name,j['reward'],model,x['step'])); hashes[str(path.relative_to(root))]=x['model_sha256']
    rows=[]
    for name,mode,model,step in specs:
        env=FiniteHorizon(TrainingEnv(loader,scaler,split='validation',initial_shares=e['initial_shares'],costs=CostConfig(**e['costs']),reward_config=RewardConfig(mode=mode,drawdown_lambda=e['drawdown_lambda'])))
        traces=[]
        try:
            for r in recs:
                frame=loader.load(r.episode_id,expected_split='validation'); seed=episode_seed(r.episode_id,cfg['experiment']['evaluation_seed'])
                if model is None:
                    b=BenchmarkPolicy(name); b.reset(float(frame.iloc[0]['Close']),seed=seed)
                    def policy(obs,close,t,h): return b.act(close=close,step=t,horizon=h)
                else:
                    require(model.observation_space==env.observation_space and model.action_space==env.action_space,'space mismatch')
                    def policy(obs,close,t,h): return int(model.predict(obs,deterministic=True)[0].item())
                m,trace=evaluate_episode(env,r,frame,name,policy,seed); rows.append(m); traces.extend(trace)
        finally: env.close()
        pd.DataFrame(traces).to_csv(tmp/'steps.csv',mode='a',header=not (tmp/'steps.csv').exists(),index=False)
        print(name,len(rows),'/',26*376,flush=True)
    df=pd.DataFrame(rows); require(len(df)==26*376 and not df.duplicated(['policy','episode_id']).any() and set(df.split)=={'validation'},'evaluation grid')
    summaries,pairs=summarize(rows); s=pd.DataFrame(summaries); b1=df[df.policy=='B1'].set_index('episode_id').terminal_return
    df['excess_b1']=df.terminal_return-df.episode_id.map(b1); df['full_exit']=df.exit_100_step.notna(); df['exit90']=df.exit_90_step.notna(); df['one_share_remaining']=df.remaining_fraction.eq(.01)
    behavior=df.groupby('policy').agg(mean_excess_b1=('excess_b1','mean'),full_exit_rate=('full_exit','mean'),exit90_rate=('exit90','mean'),one_share_rate=('one_share_remaining','mean'),mean_remaining=('remaining_fraction','mean'),mean_noop_rate=('noop_rate','mean'),mean_steps=('steps','mean')).reset_index()
    s=s.merge(behavior,on='policy',validate='one_to_one'); s[['algorithm','reward','checkpoint']]=s.policy.str.extract(r'^(PPO|A2C|DQN)_(R[12])_(\d+)$'); s['checkpoint']=pd.to_numeric(s.checkpoint)
    df.to_csv(tmp/'episodes.csv',index=False); s.to_csv(tmp/'summary.csv',index=False); pd.DataFrame(pairs).to_csv(tmp/'paired_b0.csv',index=False)
    s[s.checkpoint.notna()].sort_values(['algorithm','reward','checkpoint']).to_csv(tmp/'learning_curves.csv',index=False)
    require(verify_plan(root,plan)==inputs,'inputs changed'); require(all(digest(root/k)==v for k,v in hashes.items()),'models changed')
    write_json(tmp/'inputs.json',dict(data=inputs,models=hashes)); write_json(tmp/'run.json',dict(status='PASS',split='validation',training_run=a.training_run,plan_hash=plan['plan_hash'],policies=26,episodes=len(df),checkpoints=18,deterministic=True))
    tmp.rename(out); print('PASS',len(df),out)
if __name__=='__main__': main()
