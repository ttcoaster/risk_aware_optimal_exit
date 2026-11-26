import argparse,json,hashlib
from pathlib import Path
import pandas as pd
from src.evaluation.behavior_diagnostics import diagnose

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-run',default='validation54'); p.add_argument('--run-id',default='diagnostics55'); a=p.parse_args()
    if not a.source_run.isalnum() or not a.run_id.isalnum(): p.error('alphanumeric IDs required')
    root=Path(__file__).resolve().parents[1]/'artifacts/evaluation'; src=root/a.source_run; out=root/a.run_id; tmp=out.with_name(out.name+'.partial')
    if out.exists() or tmp.exists(): raise FileExistsError(out)
    meta=json.loads((src/'run.json').read_text());
    if meta.get('split')!='validation' or meta.get('status')!='PASS': raise ValueError('invalid source run')
    files=['episodes.csv','steps.csv','run.json','inputs.json']; hashes={f:hashlib.sha256((src/f).read_bytes()).hexdigest() for f in files}
    reports=diagnose(pd.read_csv(src/'episodes.csv',dtype={'symbol':str}),pd.read_csv(src/'steps.csv',dtype={'symbol':str}))
    tmp.mkdir(parents=True)
    for name,df in reports.items(): df.to_csv(tmp/f'{name}.csv',index=False)
    (tmp/'run.json').write_text(json.dumps(dict(source_run=a.source_run,split='validation',status='PASS',source_hashes=hashes),indent=2)); tmp.rename(out)
    print(f'PASS: {len(reports)} diagnostic tables; output={out}')
if __name__=='__main__': main()
