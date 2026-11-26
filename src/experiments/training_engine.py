"""Plan-bound train-only execution. Checkpoint scheduling belongs to STEP 6-3."""
import json
import time
from copy import deepcopy
from pathlib import Path
from .configuration import load_plan, snapshot_inputs, canonical_hash, digest, require, runtime


def verify_plan(root, plan):
    require(plan['status']=='PLAN_READY','plan status')
    expected=load_plan(root,plan['profile'])
    for k,v in expected.items():
        require(plan[k]==v,'plan mismatch: '+k)
    inputs=snapshot_inputs(root,expected['config']['environment'])
    require(inputs==plan['inputs'],'input snapshot mismatch')
    require(plan['plan_hash']==canonical_hash(dict(config_hash=expected['config_hash'],inputs=inputs)),'plan hash mismatch')
    return inputs


def select_jobs(plan, job_ids):
    require(bool(job_ids) and len(set(job_ids))==len(job_ids),'explicit unique jobs required')
    jobs={j['job_id']:j for j in plan['jobs']}
    require(set(job_ids)<=set(jobs),'unknown job')
    return [jobs[k] for k in job_ids]


def budget(job, integration):
    return 2048 if integration else job['total_timesteps']


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def train_job(root, plan, job, out, integration):
    import numpy as np
    import torch
    import stable_baselines3 as sb3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from src.data.episode_loader import EpisodeLoader
    from src.data.observation_scaler import FeatureScaler
    from src.envs.training_env import TrainingEnv
    from src.envs.finite_horizon import FiniteHorizon
    from src.envs.rewards import RewardConfig
    from src.envs.accounting import CostConfig
    root,out=Path(root),Path(out)
    inputs=verify_plan(root,plan)
    require(job in plan['jobs'],'job not in plan')
    cfg = deepcopy(plan['config'])
    e = cfg['environment']
    seed = job['seed']
    steps = budget(job, integration)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'start.json',dict(job=job,requested_steps=steps,integration=integration,plan_hash=plan['plan_hash'],inputs=inputs,runtime=runtime(root),algorithm_kwargs=cfg['algorithms'][job['algorithm']]))
    torch.set_num_threads(cfg['experiment']['torch_threads']); set_random_seed(seed)
    loader=EpisodeLoader(root/e['manifest'],root/e['feature_dir']); scaler=FeatureScaler.load(root/e['scaler'])
    raw=TrainingEnv(loader,scaler,split='train',seed=seed,initial_shares=e['initial_shares'],costs=CostConfig(**e['costs']),reward_config=RewardConfig(mode=job['reward'],drawdown_lambda=e['drawdown_lambda']))
    env=Monitor(FiniteHorizon(raw),str(out/'train.monitor.csv'))
    class Guard(BaseCallback):
        def __init__(self):
            super().__init__(); self.ids=set(); self.transitions=0
        def _on_step(self):
            for info,r,obs in zip(self.locals['infos'],self.locals['rewards'],self.locals['new_obs']):
                require(info['split']=='train' and loader.record(info['episode_id']).split=='train','nontrain transition')
                require(not info['forced_sale'] and np.isfinite(r) and env.observation_space.contains(obs),'invalid transition')
                self.ids.add(info['episode_id']); self.transitions+=1
            return True
    try:
        require(env.observation_space.shape==(9,) and env.action_space.n==5,'space mismatch')
        cls={'PPO':sb3.PPO,'A2C':sb3.A2C,'DQN':sb3.DQN}[job['algorithm']]
        model=cls(env=env,seed=seed,**cfg['algorithms'][job['algorithm']]); model.set_logger(configure(str(out),['csv']))
        before={k:v.detach().clone() for k,v in model.policy.named_parameters()}; guard=Guard(); start=time.perf_counter()
        model.learn(total_timesteps=steps,callback=guard,log_interval=1)
        elapsed=time.perf_counter()-start
        require(model.num_timesteps==steps==guard.transitions,'step count mismatch')
        delta=max(float((v.detach()-before[k]).abs().max()) for k,v in model.policy.named_parameters())
        require(np.isfinite(delta) and delta>0 and all(torch.isfinite(v).all().item() for v in model.policy.parameters()),'invalid parameter update')
        model.save(out/'model'); loaded=cls.load(out/'model.zip',device='cpu')
        require(loaded.num_timesteps==steps,'reload step mismatch')
        probes=0
        probe=FiniteHorizon(TrainingEnv(loader,scaler,split='train',seed=seed,initial_shares=e['initial_shares'],costs=CostConfig(**e['costs']),reward_config=RewardConfig(mode=job['reward'],drawdown_lambda=e['drawdown_lambda'])))
        for rec in loader.records('train')[:10]:
            obs,_=probe.reset(seed=seed,options={'episode_id':rec.episode_id})
            for _ in range(rec.decision_steps):
                a,_=model.predict(obs,deterministic=True); b,_=loaded.predict(obs,deterministic=True)
                require(int(a)==int(b) and env.action_space.contains(int(a)),'reload action mismatch')
                obs,r,term,trunc,info=probe.step(int(a)); probes+=1
                require(info['split']=='train' and not info['forced_sale'] and np.isfinite(r),'invalid probe')
                if term or trunc: break
            require(term and not trunc,'probe terminal flags')
        probe.close()
        require(verify_plan(root,plan)==inputs,'post-training inputs')
        result=dict(status='PASS',job_id=job['job_id'],integration=integration,requested_steps=steps,actual_steps=model.num_timesteps,train_transitions=guard.transitions,unique_train_episodes=len(guard.ids),max_parameter_change=delta,reload_probe_steps=probes,elapsed_seconds=elapsed,input_hashes_verified=True,model_sha256=digest(out/'model.zip'),plan_hash=plan['plan_hash'])
        write_json(out/'summary.json',result); return result
    finally:
        env.close()
