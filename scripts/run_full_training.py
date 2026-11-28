import argparse,json
from pathlib import Path
from src.experiments.full_training import read,train

def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan-id',required=True); p.add_argument('--run-id',required=True); a=p.parse_args()
    if not all(x.isascii() and x.isalnum() for x in vars(a).values()): p.error('ASCII alphanumeric IDs required')
    root=Path(__file__).resolve().parents[1]; plan=read(root/'artifacts/experiment_plans'/a.plan_id/'plan.json')
    print(json.dumps(train(root,plan,root/'artifacts/tracked_training'/a.run_id),indent=2))
if __name__=='__main__': main()
