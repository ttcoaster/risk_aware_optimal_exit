# STEP 5-2
Copy files into project root. No existing files modified. Run:
```
python -m unittest discover -s tests -p test_metrics.py -v
python -m unittest discover -s tests -p "test_*.py" -v
python -m scripts.audit_metrics
```
Metrics use close-to-close portfolio marks, zero cash interest, flat padding only after full exit. MDD positive; daily Sharpe/Sortino unannualized, zero target, undefined ratios None. Exposure averages start-of-step stock weights. No-op denominator non-hold requests; cancelled orders included. Costs inferred from next open gross less cash increment, including slippage. CEQ uses population variance of fixed-horizon simple returns, gamma=3 illustrative. Empirical VaR/CVaR positive-loss convention (can be negative), fractional tail mass, alpha=.05. Eight train episodes are smoke only, not statistical evidence. No test data, no model training. Paired excess uses matched episode IDs. Overlapping episodes not independent. Adjusted prices and no dividends remain existing assumptions.
