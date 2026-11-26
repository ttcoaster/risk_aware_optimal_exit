"""Read-only STEP 4-1 audit. Run: python -m scripts.audit_episode_loader"""
from pathlib import Path
from collections import Counter
import pandas as pd
from src.data.episode_loader import EpisodeLoader, EpisodeSampler, manifest_overlap_report, SPLITS


def main():
    root = Path(__file__).resolve().parents[1]
    loader = EpisodeLoader(root / 'data/processed/episodes/episode_manifest.csv', root / 'data/processed/s1')
    print('STEP 4-1 READ-ONLY EPISODE AUDIT')
    for split, stats in manifest_overlap_report(loader).items():
        print(split, stats)
    ids = EpisodeSampler(loader, 'train', seed=42).sample_ids(5)
    print('seed=42 train samples:', ids)
    assert ids == EpisodeSampler(loader, 'train', seed=42).sample_ids(5)
    assert not set(r.episode_id for r in loader.records('train')) & set(r.episode_id for r in loader.records('test'))
    for split in SPLITS:
        count = 0
        for record in loader.records(split):
            frame = loader.load(record.episode_id, expected_split=split)
            assert len(frame) == record.decision_steps + 1
            count += 1
        print(f'{split}: validated {count} windows')
    print('PASS: all windows, split guard, reproducible sampling')
    print('CAUTION: overlap is expected; this does not validate dividends/splits or economic total returns.')


if __name__ == '__main__':
    main()
