import argparse,json
from pathlib import Path
from src.evaluation.integrity_report import audit

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-run',default='validation54'); p.add_argument('--diagnostic-run',default='diagnostics55'); p.add_argument('--run-id',default='integrity56'); a=p.parse_args()
    if not all(x.isalnum() for x in vars(a).values()): p.error('alphanumeric IDs required')
    root=Path(__file__).resolve().parents[1]; base=root/'artifacts/evaluation'; out=base/a.run_id
    if out.exists(): raise FileExistsError(out)
    result=audit(root,base/a.source_run,base/a.diagnostic_run); out.mkdir(); (out/'audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
