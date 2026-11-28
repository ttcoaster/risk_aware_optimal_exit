"""Full-training-only training and validation guards."""
import json
from pathlib import Path
from .configuration import require, digest
from .training_engine import verify_plan, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def full_jobs(plan):
    require(plan['profile']=='full_training','full profile required')
    jobs=plan['jobs']
    require(len(jobs)==30 and {(j['algorithm'],j['reward'],j['seed']) for j in jobs}=={(a,r,s) for a in ('PPO','A2C','DQN') for r in ('R1','R2') for s in (42,7,17,29,53)},'full grid')
    require(all(j['total_timesteps']==102400 and j['checkpoints']==[20480,51200,102400] for j in jobs),'full budget')
    return jobs


def checkpoints(plan, folder):
    jobs=full_jobs(plan); folder=Path(folder); summary=read(folder/'summary.json')
    require(read(folder/'plan.json')==plan,'saved plan mismatch')
    require(summary['status']=='PASS' and summary['integration'] is False and summary['jobs']==30 and summary['checkpoints']==90 and summary['total_steps']==3072000,'full summary')
    require(len(summary['results'])==30,'summary grid')
    results={r['job_id']:r for r in summary['results']}; output=[]
    for j in jobs:
        name=j['job_id']; base=folder/name; r=read(base/'summary.json')
        require(r==results[name] and r['status']=='PASS' and r['integration'] is False and r['actual_steps']==r['train_transitions']==r['requested_steps']==102400 and r['input_hashes_verified'] is True and r['plan_hash']==plan['plan_hash'],'job summary')
        rows=read(base/'checkpoints.json')
        require([x['step'] for x in rows]==j['checkpoints']==r['checkpoint_steps'],'checkpoint grid')
        for x in rows:
            path=base/'checkpoints'/f"step_{x['step']:08d}"/'model.zip'
            require(x['path']==path.relative_to(base).as_posix() and x['job_id']==name and x['plan_hash']==plan['plan_hash'],'checkpoint identity')
            require(read(path.parent/'checkpoint.json')==x and digest(path)==x['model_sha256'],'checkpoint integrity')
            output.append((j,x,path))
        require(rows[-1]['parameter_sha256']==r['parameter_sha256'],'final fingerprint')
    return output


def train(root,plan,out):
    from .tracked_training import train_tracked_job
    verify_plan(root,plan); jobs=full_jobs(plan); out=Path(out)
    out.mkdir(parents=True,exist_ok=False); write_json(out/'plan.json',plan)
    results=[]
    for j in jobs:
        r=train_tracked_job(root,plan,j,out/j['job_id'],False); results.append(r); print(json.dumps(r),flush=True)
    verify_plan(root,plan)
    summary=dict(status='PASS',integration=False,jobs=len(results),checkpoints=sum(len(r['checkpoint_steps']) for r in results),total_steps=sum(r['actual_steps'] for r in results),results=results)
    write_json(out/'summary.json',summary); checkpoints(plan,out); return summary
