from pathlib import Path
import numpy as np
from src.data.episode_loader import EpisodeLoader
from src.data.observation_scaler import FeatureScaler
from src.envs.training_env import TrainingEnv
from src.envs.rewards import RewardConfig
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    root=Path(__file__).resolve().parents[1]
    loader=EpisodeLoader(root/'data/processed/episodes/episode_manifest.csv',root/'data/processed/s1')
    scaler=FeatureScaler.load(root/'artifacts/normalization/s1_train_scaler.json')
    for mode in ('R1','R2'):
        env=TrainingEnv(loader,scaler,reward_config=RewardConfig(mode=mode))
        vec=DummyVecEnv([lambda:env]); vec.seed(42); obs=vec.reset(); ids=[]; steps=0
        for _ in range(10):
            eid=env.episode_id; ids.append(eid)
            while True:
                obs,reward,done,infos=vec.step(np.array([0]))
                steps+=1; info=infos[0]
                assert info['split']=='train' and info['episode_id']==eid
                assert env.observation_space.contains(obs[0])
                if done[0]:
                    assert env.observation_space.contains(info['terminal_observation'])
                    assert info['shares_remaining']==100 and not info['forced_sale']
                    assert obs[0,8]==0 and obs[0,6]==1
                    break
        print(f'{mode}: episodes=10, steps={steps}, unique_ids={len(set(ids))}, shape={obs.shape}')
        vec.close()
    print('PASS: split-safe resets, 9D observations, SB3 automatic reset')
if __name__=='__main__': main()
