"""Explicit jobs only; integration mode is intentionally separate from research runs."""
import argparse,json
from pathlib import Path
from src.experiments.training_engine import verify_plan,select_jobs,train_job,write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan-id',required=True); p.add_argument('--run-id',required=True); p.add_argument('--jobs',nargs='+',required=True); p.add_argument('--integration',action='store_true'); a=p.parse_args()
    if not all(x.isascii() and x.isalnum() for x in (a.plan_id,a.run_id)): p.error('ASCII alphanumeric IDs required')
    if not a.integration: p.error('STEP 6-2 permits integration runs only; full training follows STEP 6-3')
    root=Path(__file__).resolve().parents[1]; plan=json.loads((root/'artifacts/experiment_plans'/a.plan_id/'plan.json').read_text(encoding='utf-8'))
    verify_plan(root,plan); jobs=select_jobs(plan,a.jobs)
    out=root/'artifacts/training_engine'/a.run_id; out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan); results=[]
    for job in jobs:
        result=train_job(root,plan,job,out/job['job_id'],a.integration); results.append(result); print(json.dumps(result),flush=True)
    summary=dict(status='PASS',integration=True,jobs=len(results),total_steps=sum(r['actual_steps'] for r in results),results=results)
    write_json(out/'summary.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
