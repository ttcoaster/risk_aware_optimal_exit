"""Validation-only audit; never trains or loads a model."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.behavior_diagnostics import diagnose
from src.evaluation.metrics import StepRecord,episode_metrics

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise ValueError(msg)
def read(p): return pd.read_csv(p,dtype={'symbol':str})
def audit(root,source,diagnostics,verify_inputs=True):
    root,source,diagnostics=map(Path,(root,source,diagnostics))
    meta=json.loads((source/'run.json').read_text()); dm=json.loads((diagnostics/'run.json').read_text())
    require(meta['split']==dm['split']=='validation' and meta['status']==dm['status']=='PASS' and dm['source_run']==source.name and meta['deterministic'] is True,'run metadata')
    require(set(dm['source_hashes'])=={'episodes.csv','steps.csv','run.json','inputs.json'},'hash coverage')
    for name,h in dm['source_hashes'].items(): require(digest(source/name)==h,'source hash: '+name)
    inputs=json.loads((source/'inputs.json').read_text())
    if verify_inputs:
        require(len(inputs)==16,'input coverage')
        for name,h in inputs.items():
            p=(root/name.replace('\\','/')).resolve(); require(p.is_relative_to(root.resolve()),'unsafe path'); require(digest(p)==h,'input hash: '+name)
    e,s=read(source/'episodes.csv'),read(source/'steps.csv'); reports=diagnose(e,s)
    for name,df in reports.items():
        actual=read(diagnostics/(name+'.csv')); keys=[c for c in ['policy','symbol','year','market','holding_band','action','step'] if c in df]
        for x in (actual,df):
            for k in keys: x[k]=x[k].astype(str)
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),df.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-9,atol=1e-10)
    manifest=read(root/'data/processed/episodes/episode_manifest.csv'); m=manifest[manifest.split.eq('validation')].set_index('episode_id'); require(len(m)==376 and set(m.index)==set(e.episode_id),'manifest grid')
    frames={sym:pd.read_parquet(root/'data/processed/s1'/f'{sym}_s1.parquet') for sym in m.symbol.unique()}
    indexed=e.set_index(['policy','episode_id']); max_error=0.
    for (policy,eid),g in s.groupby(['policy','episode_id']):
        r=indexed.loc[policy,eid]; rec=m.loc[eid]; g=g.sort_values('step'); f=frames[rec.symbol].loc[rec.start_date:rec.last_execution_date]; n=len(g)
        require(len(f)==61 and r.symbol==rec.symbol and g.symbol.eq(rec.symbol).all(),'market window')
        require(np.array_equal(pd.to_datetime(g.decision_date).to_numpy(),pd.to_datetime(f.index[:n]).to_numpy()) and np.array_equal(pd.to_datetime(g.execution_date).to_numpy(),pd.to_datetime(f.index[1:n+1]).to_numpy()),'date alignment')
        require(np.array_equal(g.execution_allowed.to_numpy(),f.Tradable.iloc[1:n+1].to_numpy()),'tradability')
        gross=g.shares_sold.to_numpy()*f.Open.iloc[1:n+1].to_numpy(); cost=gross*(1-(1-.001)*(1-.001-.002)); cash=np.cumsum(gross-cost); value=cash+g.shares.to_numpy()*f.Close.iloc[1:n+1].to_numpy(); initial=100*float(f.Close.iloc[0]); values=np.r_[initial,value]; dd=1-values/np.maximum.accumulate(values); penalty=.5*np.maximum(np.diff(dd),0) if policy.endswith('_R2') else np.zeros(n)
        for a,b in [(g.cost,cost),(g.cash,cash),(g.value,value),(g.risk_penalty,penalty),(g.reward,np.diff(np.log(values))-penalty)]:
            require(np.allclose(a,b,rtol=1e-10,atol=1e-7),'market accounting: '+eid+' '+policy); max_error=max(max_error,float(np.max(np.abs(a-b))))
        metrics=episode_metrics(initial,100,[StepRecord(float(x.value),int(x.shares),float(x.cash),int(x.action),bool(x.execution_allowed),float(x.cost)) for x in g.itertuples()],60)
        for k,v in metrics.items(): require(pd.isna(r[k]) if v is None else np.isclose(r[k],v,rtol=1e-9,atol=1e-8),'metric: '+k)
    return dict(status='PASS',episodes=len(e),steps=len(s),diagnostic_tables=len(reports),source_hashes=dm['source_hashes'],input_hashes_verified=verify_inputs,max_absolute_accounting_error=max_error,limitations=['No model inference replay','No proof of training-time parquet identity','No statistical inference'])
