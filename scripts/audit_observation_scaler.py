"""Read-only real data audit. Saves only fitted TRAIN statistics to artifacts/."""
from pathlib import Path
import numpy as np
from src.data.episode_loader import EpisodeLoader
from src.data.observation_scaler import fit_train_scaler
from src.envs.normalized_observation import NormalizedObservation


def main():
    root=Path(__file__).resolve().parents[1]
    loader=EpisodeLoader(root/'data/processed/episodes/episode_manifest.csv',root/'data/processed/s1')
    scaler,counts=fit_train_scaler(loader)
    print('STEP 4-2 OBSERVATION AUDIT')
    print('train unique valid rows by symbol:',counts)
    print('total unique train rows:',sum(counts.values()))
    print('mean:',scaler.mean)
    print('scale:',scaler.scale)
    for split in ('train','validation','test'):
        records=loader.records(split)
        clipped=0;total=0
        for rec in records:
            # Inspect all decision rows; count repeated windows here as a diagnostic only.
            frame=loader.load(rec.episode_id,expected_split=split)
            valid=frame.iloc[:-1]
            valid=valid.loc[valid['FeatureReady'].astype(bool)]
            if len(valid):
                x=valid[list(('return_1d','return_5d','return_20d','volatility_20d','ma_gap_20d'))].to_numpy(dtype=float)
                z=(x-np.asarray(scaler.mean))/np.asarray(scaler.scale)
                clipped+=int((np.abs(z)>scaler.clip).sum());total+=int(z.size)
        print(f'{split}: episodes={len(records)}, clipped_feature_values={clipped}/{total}')
    # Real Episode smoke: wrapper must preserve portfolio accounting and reward.
    rec=loader.records('test')[0]
    raw=loader.make_env(rec.episode_id,expected_split='test')
    wrapped=NormalizedObservation(loader.make_env(rec.episode_id,expected_split='test'),scaler)
    o1,_=raw.reset(seed=42);o2,_=wrapped.reset(seed=42)
    assert np.allclose(o1[5:],o2[5:])
    assert wrapped.observation_space.contains(o2)
    for i in range(rec.decision_steps):
        action=1 if i==0 else 0
        a=raw.step(action);b=wrapped.step(action)
        assert np.isclose(a[1],b[1]) and a[2:4]==b[2:4]
        assert np.isclose(a[4]['portfolio_value'],b[4]['portfolio_value'])
        assert wrapped.observation_space.contains(b[0])
        if a[2] or a[3]: break
    output=root/'artifacts/normalization/s1_train_scaler.json'
    scaler.save(output)
    print('scaler saved:',output)
    print('PASS: train-only fit, bounded observations, identical rewards/accounting')

if __name__=='__main__':main()
