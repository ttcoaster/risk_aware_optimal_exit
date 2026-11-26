"""Resolve a frozen E1 experiment plan; no training or model inference."""
import csv
import hashlib
import importlib.metadata as metadata
import itertools
import json
import platform
import subprocess
from pathlib import Path
import yaml

ALGORITHMS = ('PPO', 'A2C', 'DQN')
SYMBOLS = {'000270','000660','005380','005930','035420','035720','055550','105560'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_yaml(path):
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node):
        result = {}
        for k, v in node.value:
            key = loader.construct_object(k)
            if key in result: raise ValueError(f'duplicate YAML key: {key}')
            result[key] = loader.construct_object(v)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    obj = yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader)
    if not isinstance(obj, dict): raise ValueError('expected mapping')
    return obj

def require(condition, message):
    if not condition: raise ValueError(message)

def integer(x):
    return type(x) is int

def validate(env, exp, algs):
    # v1 intentionally freezes environment and smoke hyperparameters; extensions need schema review.
    expected = dict(schema_version=1,state='S1',observation_dim=9,action_fractions=[0,.25,.5,.75,1],horizon=60,initial_shares=100,costs=dict(commission_rate=.001,sell_tax_rate=.002,slippage_rate=.001),drawdown_lambda=.5,train_split='train',evaluation_split='validation',manifest='data/processed/episodes/episode_manifest.csv',scaler='artifacts/normalization/s1_train_scaler.json',feature_dir='data/processed/s1')
    require(env == expected,'environment differs from frozen E1 v1')
    require(set(exp)=={'schema_version','algorithms','rewards','seeds','total_timesteps','checkpoints','evaluation_seed','torch_threads'} and exp['schema_version']==1,'experiment schema')
    for key, allowed in [('algorithms',ALGORITHMS),('rewards',('R1','R2'))]:
        x=exp[key]; require(isinstance(x,list) and bool(x) and all(v in allowed for v in x) and len(set(x))==len(x),key)
    seeds=exp['seeds']; require(isinstance(seeds,list) and bool(seeds) and all(integer(s) and 0<=s<2**32 for s in seeds) and len(set(seeds))==len(seeds),'seeds')
    require(integer(exp['evaluation_seed']) and 0<=exp['evaluation_seed']<2**32,'evaluation seed')
    require(integer(exp['torch_threads']) and exp['torch_threads']>0,'torch threads')
    n=exp['total_timesteps']; c=exp['checkpoints']
    require(integer(n) and n>=2048 and isinstance(c,list) and bool(c) and all(integer(t) and t>=2048 for t in c) and c==sorted(set(c)) and c[-1]==n,'checkpoint schedule')
    require(set(algs)==set(exp['algorithms']),'algorithm set')
    common=dict(policy='MlpPolicy',device='cpu',gamma=1.,learning_rate=.0003,verbose=0,policy_kwargs=dict(net_arch=[64,64]))
    extra=dict(PPO=dict(n_steps=128,batch_size=64,n_epochs=4,gae_lambda=.95),A2C=dict(n_steps=16,gae_lambda=.95),DQN=dict(buffer_size=10000,learning_starts=128,batch_size=64,train_freq=4,gradient_steps=1,target_update_interval=250,exploration_fraction=.5))
    for name, params in algs.items():
        require(params==dict(common,**extra[name]),f'{name} differs from frozen smoke parameters')
        unit=params.get('n_steps',params.get('train_freq',1))
        require(all(t%unit==0 for t in c),f'{name} checkpoint alignment')

def load_plan(root, profile):
    require(profile in ('pilot','full_training'),'unknown profile')
    root=Path(root); base=root/'configs'
    paths=[base/'environment/optimal_exit.yaml',base/f'experiments/{profile}.yaml']
    env,exp=map(read_yaml,paths)
    names=exp.get('algorithms',[])
    require(isinstance(names,list) and all(n in ALGORITHMS for n in names),'algorithm names')
    algs={}
    for n in names:
        p=base/f'algorithms/{n.lower()}.yaml'; paths.append(p); algs[n]=read_yaml(p)
    validate(env,exp,algs)
    config=dict(environment=env,experiment=exp,algorithms=algs)
    jobs=[dict(job_id=f'{a}_{r}_seed{s}',algorithm=a,reward=r,seed=s,total_timesteps=exp['total_timesteps'],checkpoints=exp['checkpoints']) for a,r,s in itertools.product(names,exp['rewards'],exp['seeds'])]
    return dict(profile=profile,config=config,config_hash=canonical_hash(config),config_files={p.relative_to(root).as_posix():digest(p) for p in paths},jobs=jobs)

def snapshot_inputs(root, env):
    root=Path(root); manifest=root/env['manifest']; scaler=root/env['scaler']
    counts={}; symbols=set(); ids=set()
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            require(r['episode_id'] not in ids,'duplicate episode'); ids.add(r['episode_id'])
            require(r['split'] in ('train','validation','test') and int(r['decision_steps'])==60,'manifest split/horizon')
            counts[r['split']]=counts.get(r['split'],0)+1; symbols.add(r['symbol'])
    require(counts==dict(train=1310,validation=376,test=360) and symbols==SYMBOLS,'unexpected manifest universe')
    paths=[manifest,scaler]+[root/env['feature_dir']/f'{s}_s1.parquet' for s in sorted(symbols)]
    return dict(split_counts=counts,sha256={p.relative_to(root).as_posix():digest(p) for p in paths})

def runtime(root):
    def git(*args):
        try: return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError): return None
    versions={}
    for p in ('numpy','pandas','torch','gymnasium','stable-baselines3','pyarrow','fastparquet','PyYAML'):
        try: versions[p]=metadata.version(p)
        except metadata.PackageNotFoundError: versions[p]=None
    return dict(python=platform.python_version(),platform=platform.platform(),packages=versions,git_commit=git('rev-parse','HEAD'),git_status=git('status','--porcelain','--untracked-files=normal'))
