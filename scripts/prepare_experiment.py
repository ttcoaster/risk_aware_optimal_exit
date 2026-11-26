"""Write a plan only; does not launch learning."""
import argparse
import json
from pathlib import Path
from src.experiments.configuration import load_plan,snapshot_inputs,runtime,canonical_hash

def main():
    p=argparse.ArgumentParser(); p.add_argument('--profile',choices=['pilot','full_training'],required=True); p.add_argument('--run-id',required=True); a=p.parse_args()
    if not a.run_id.isascii() or not a.run_id.isalnum(): p.error('ASCII alphanumeric run ID required')
    root=Path(__file__).resolve().parents[1]; out=root/'artifacts/experiment_plans'/a.run_id
    if out.exists(): raise FileExistsError(out)
    plan=load_plan(root,a.profile); inputs=snapshot_inputs(root,plan['config']['environment']); rt=runtime(root)
    plan.update(run_id=a.run_id,status='PLAN_READY',inputs=inputs,runtime=rt,plan_hash=canonical_hash(dict(config_hash=plan['config_hash'],inputs=inputs)))
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=plan['status'],jobs=len(plan['jobs']),total_timesteps=sum(j['total_timesteps'] for j in plan['jobs']),input_files=len(inputs['sha256']),split_counts=inputs['split_counts'],config_hash=plan['config_hash'],plan_hash=plan['plan_hash'],output=str(out)),indent=2))
if __name__=='__main__': main()
