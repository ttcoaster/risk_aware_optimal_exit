import unittest
import numpy as np
import pandas as pd
from src.data.build_episode_manifest import build_manifest_for_symbol


def sample(n=160):
    idx = pd.bdate_range('2018-09-03',periods=n)
    d = pd.DataFrame(index=idx)
    for c in ('Open','High','Low','Close'):
        d[c] = 100.0
    d['Volume'] = 100
    d['Tradable'] = True
    d['FeatureReady'] = True
    for c in ('return_1d','return_5d','return_20d','volatility_20d','ma_gap_20d'):
        d[c] = 0.0
    return d

class TestEpisodeManifest(unittest.TestCase):
    def test_60_decisions_need_61_rows(self):
        d=sample(61)
        s={'train':('2018-01-01','2019-12-31')}
        rows=build_manifest_for_symbol(d,'005930',horizon=60,stride=20,splits=s)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['last_decision_date'],str(d.index[59].date()))
        self.assertEqual(rows[0]['last_execution_date'],str(d.index[60].date()))
        self.assertEqual(build_manifest_for_symbol(d.iloc[:60],'005930',horizon=60,splits=s),[])

    def test_no_cross_split(self):
        d=sample(160)
        splits={'train':('2018-01-01','2018-12-31'), 'validation':('2019-01-01','2019-12-31')}
        rows=build_manifest_for_symbol(d,'005930',horizon=60,stride=20,splits=splits)
        for r in rows:
            end='2018-12-31' if r['split']=='train' else '2019-12-31'
            self.assertLessEqual(r['last_execution_date'],end)

    def test_suspension_inside_episode_retained(self):
        d=sample(65)
        d.loc[d.index[10],'Tradable']=False
        d.loc[d.index[10],'FeatureReady']=False
        s={'train':('2018-01-01','2019-12-31')}
        rows=build_manifest_for_symbol(d,'005930',horizon=60,stride=20,splits=s)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['nontrading_decision_rows'],1)

    def test_entry_requires_ready(self):
        d=sample(61)
        d.loc[d.index[0],'FeatureReady']=False
        s={'train':('2018-01-01','2019-12-31')}
        self.assertEqual(build_manifest_for_symbol(d,'005930',horizon=60,splits=s),[])

    def test_unsorted_rejected(self):
        d=sample(61).iloc[::-1]
        with self.assertRaises(ValueError):
            build_manifest_for_symbol(d,'005930')

if __name__=='__main__':
    unittest.main()
