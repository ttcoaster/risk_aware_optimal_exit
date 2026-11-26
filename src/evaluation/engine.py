"""Shared, read-only evaluation for E1 policies."""
import hashlib
import math
from dataclasses import asdict
from src.evaluation.metrics import StepRecord, episode_metrics, distribution_metrics, paired_excess


def episode_seed(eid, seed=42):
    return int.from_bytes(hashlib.sha256(f'{seed}:{eid}'.encode()).digest()[:4], 'big')


def evaluate_episode(env, rec, frame, policy_id, policy, seed=42):
    """policy(obs, current_close, step, horizon) -> discrete action."""
    obs, info = env.reset(seed=seed, options={'episode_id':rec.episode_id})
    initial = info['initial_value']; shares=100; cash=0.; rows=[]; trace=[]
    total=penalty=0.
    for t in range(rec.decision_steps):
        action=int(policy(obs, float(frame.iloc[t]['Close']), t, rec.decision_steps))
        if action not in range(5): raise ValueError('invalid action')
        obs, reward, term, trunc, info=env.step(action)
        if info['episode_id']!=rec.episode_id or info['split']!=rec.split or info['forced_sale'] or not env.observation_space.contains(obs):
            raise RuntimeError('invalid transition')
        sold=info['shares_sold']; remaining=info['shares_remaining']; new_cash=info['cash']
        if shares-sold!=remaining or (sold and not info['execution_allowed']): raise RuntimeError('invalid execution')
        cost=sold*float(frame.iloc[t+1]['Open'])-(new_cash-cash) if sold else 0.
        if cost < -1e-7: raise RuntimeError('negative cost')
        row=StepRecord(info['portfolio_value'],remaining,new_cash,action,bool(info['execution_allowed']),max(0.,cost))
        rows.append(row); total+=float(reward); penalty+=float(info['risk_penalty'])
        trace.append(dict(policy=policy_id,episode_id=rec.episode_id,symbol=rec.symbol,split=rec.split,step=t+1,decision_date=str(frame.index[t]),execution_date=str(frame.index[t+1]),shares_sold=sold,reward=float(reward),risk_penalty=info['risk_penalty'],**asdict(row)))
        shares,cash=remaining,new_cash
        if term or trunc: break
    if not term or trunc: raise RuntimeError('invalid terminal flags')
    m=episode_metrics(initial,100,rows,rec.decision_steps)
    if not math.isclose(total,m['log_return']-penalty,abs_tol=1e-8) or not math.isclose(info['terminal_value'],m['terminal_value'],abs_tol=1e-8): raise RuntimeError('accounting mismatch')
    return dict(policy=policy_id,episode_id=rec.episode_id,symbol=rec.symbol,split=rec.split,seed=seed,reward_sum=total,risk_penalty=penalty,**m),trace


def summarize(rows):
    groups={}
    for r in rows:
        g=groups.setdefault(r['policy'],{})
        if r['episode_id'] in g: raise ValueError('duplicate episode')
        g[r['episode_id']]=r
    if 'B0' not in groups: raise ValueError('B0 required')
    base={k:r['terminal_return'] for k,r in groups['B0'].items()}
    summaries=[]; pairs=[]
    for name,g in groups.items():
        excess=paired_excess({k:r['terminal_return'] for k,r in g.items()},base)
        for k,v in excess.items(): pairs.append(dict(policy=name,episode_id=k,benchmark='B0',excess_return=v,delta_mdd=g[k]['mdd']-groups['B0'][k]['mdd']))
        summaries.append(dict(policy=name,**distribution_metrics([r['terminal_return'] for r in g.values()]),mean_mdd=sum(r['mdd'] for r in g.values())/len(g),mean_exposure=sum(r['mean_stock_weight'] for r in g.values())/len(g),mean_excess=sum(excess.values())/len(g)))
    return summaries,pairs
