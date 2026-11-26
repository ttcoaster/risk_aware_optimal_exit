"""Descriptive paired diagnostics; no inference or model selection."""
from statistics import mean
from src.evaluation.metrics import distribution_metrics

def diagnostics(rows, expected_ids, policies):
    groups={p:{} for p in policies}
    for r in rows:
        p,e=r['policy'],r['episode_id']
        if r['split']!='validation' or p not in groups or e in groups[p]:
            raise ValueError('invalid policy/episode/split')
        groups[p][e]=r
    if not expected_ids or any(set(g)!=expected_ids for g in groups.values()):
        raise ValueError('incomplete paired grid')
    behavior=[]; pairs=[]; market=[]
    for p,g in groups.items():
        rs=list(g.values()); n=len(rs)
        behavior.append(dict(policy=p,n=n,immediate_rate=mean(r['exit_100_step']==1 for r in rs),exit90_rate=mean(r['exit_90_step'] is not None for r in rs),full_exit_rate=mean(r['exit_100_step'] is not None for r in rs),mean_remaining=mean(r['remaining_fraction'] for r in rs),mean_exposure=mean(r['mean_stock_weight'] for r in rs),noop_rate=sum(r['sell_requests']-r['fills'] for r in rs)/sum(r['sell_requests'] for r in rs) if sum(r['sell_requests'] for r in rs) else None,mean_cost_fraction=mean(r['cost_fraction'] for r in rs)))
        for label, predicate in [('up',lambda x:x>0),('down',lambda x:x<0),('flat',lambda x:x==0)]:
            subset=[r for e,r in g.items() if predicate(groups['B0'][e]['terminal_return'])]
            if subset:
                market.append(dict(policy=p,market=label,**distribution_metrics([r['terminal_return'] for r in subset]),mean_mdd=mean(r['mdd'] for r in subset),mean_exposure=mean(r['mean_stock_weight'] for r in subset),mean_excess=mean(r['terminal_return']-groups['B0'][r['episode_id']]['terminal_return'] for r in subset)))
    for alg in ('PPO','A2C','DQN'):
        for e in sorted(expected_ids):
            a,b=groups[alg+'_R1'][e],groups[alg+'_R2'][e]
            pairs.append(dict(algorithm=alg,episode_id=e,symbol=a['symbol'],delta_return=b['terminal_return']-a['terminal_return'],delta_mdd=b['mdd']-a['mdd'],delta_exposure=b['mean_stock_weight']-a['mean_stock_weight']))
    return behavior,pairs,market
