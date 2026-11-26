"""Read-only diagnostics of recorded validation trajectories."""
import numpy as np
import pandas as pd
KEY=['policy','episode_id']
POLICIES={f'B{i}' for i in range(8)}|{f'{a}_{r}' for a in ('PPO','A2C','DQN') for r in ('R1','R2')}

def diagnose(e,s):
    e=e.copy(); s=s.copy()
    if set(e.split)!= {'validation'} or set(s.split)!= {'validation'}: raise ValueError('validation only')
    if e.duplicated(KEY).any() or s.duplicated(KEY+['step']).any(): raise ValueError('duplicate')
    if set(e.policy)!=POLICIES or e.episode_id.nunique()!=376 or len(e)!=5264 or not e.groupby('episode_id').policy.nunique().eq(14).all(): raise ValueError('incomplete grid')
    s=s.sort_values(KEY+['step']); g=s.groupby(KEY); a=e.set_index(KEY).sort_index()
    if not g.size().sort_index().equals(a.steps.astype('int64')) or not s.step.eq(g.cumcount()+1).all(): raise ValueError('step mismatch')
    s['before']=g.shares.shift().fillna(100)
    if not s.before.sub(s.shares_sold).eq(s.shares).all(): raise ValueError('share mismatch')
    if not s.action.isin(range(5)).all() or not s.execution_allowed.isin([True,False]).all(): raise ValueError('invalid action or flag')
    expected=np.floor(s.before*s.action.map({0:0,1:.25,2:.5,3:.75,4:1}))*s.execution_allowed
    if not expected.eq(s.shares_sold).all(): raise ValueError('execution mismatch')
    if not np.allclose(g.shares.last().sort_index(),a.remaining_fraction*100) or not np.allclose(g.value.last().sort_index(),a.terminal_value): raise ValueError('terminal mismatch')
    if not np.allclose(g.reward.sum().sort_index(),a.reward_sum,atol=1e-9) or not np.allclose(g.risk_penalty.sum().sort_index(),a.risk_penalty,atol=1e-9) or not np.allclose(a.reward_sum+a.risk_penalty,a.log_return,atol=1e-9): raise ValueError('reward mismatch')
    s['request']=s.action.gt(0); s['fill']=s.shares_sold.gt(0); s['cancel']=s.request & ~s.execution_allowed; s['rounding']=s.request & s.execution_allowed & ~s['fill']; s['one_share']=s.before.eq(1); s['one_share_noop']=s.one_share & s.rounding
    s['holding_band']=pd.cut(s.before,[0,1,4,10,50,100],labels=['1','2-4','5-10','11-50','51-100'])
    cols=['request','fill','cancel','rounding','one_share','one_share_noop']
    d=s.groupby('policy')[cols].sum(); d['steps']=s.groupby('policy').size(); d['noop_rate']=(d.cancel+d.rounding)/d.request.replace(0,np.nan)
    actions=s.groupby(['policy','action']).agg(count=('step','size'),fills=('fill','sum'),rounding=('rounding','sum'),cancel=('cancel','sum')).reset_index()
    bands=s.groupby(['policy','holding_band','action'],observed=True).agg(count=('step','size'),fills=('fill','sum'),rounding=('rounding','sum')).reset_index()
    base=e[e.policy.eq('B0')].set_index('episode_id').terminal_return; e['market']=e.episode_id.map(base).map(lambda x:'up' if x>0 else 'down' if x<0 else 'flat'); e['year']=e.episode_id.str[-8:-4]; e['immediate']=e.exit_100_step.eq(1); e['full_exit']=e.exit_100_step.notna(); e['exit90']=e.exit_90_step.notna()
    e['excess_b1']=e.terminal_return-e.episode_id.map(e[e.policy.eq('B1')].set_index('episode_id').terminal_return)
    def group(keys): return e.groupby(keys).agg(n=('episode_id','size'),mean_return=('terminal_return','mean'),mean_mdd=('mdd','mean'),mean_exposure=('mean_stock_weight','mean'),immediate_rate=('immediate','mean'),exit90_rate=('exit90','mean'),full_exit_rate=('full_exit','mean'),mean_exit90_conditional=('exit_90_step','mean'),mean_remaining=('remaining_fraction','mean'),mean_excess_b1=('excess_b1','mean')).reset_index()
    survival=[]
    for name,x in e.groupby('policy'):
        for t in range(1,61): survival.append(dict(policy=name,step=t,exit90_rate=x.exit_90_step.le(t).mean(),full_exit_rate=x.exit_100_step.le(t).mean()))
    return {'execution_diagnostics':d.reset_index(),'action_counts':actions,'holding_action_counts':bands,'policy_behavior':group(['policy']),'market_behavior':group(['policy','market']),'symbol_behavior':group(['policy','symbol']),'year_behavior':group(['policy','year']),'exit_curves':pd.DataFrame(survival)}
