# STEP 5-1 — Rule-based benchmarks
Copy the three files into the project root preserving paths. No existing files need replacement. Run from project root in optimal-risk-exit:

    python -m unittest discover -s tests -p test_benchmarks.py -v
    python -m unittest discover -s tests -p "test_*.py" -v
    python -m scripts.audit_benchmarks

12 new unit tests. Audit uses first train episode per symbol, all eight policies, R1 default, same environment and illustrative costs as STEP 4-4. No validation/test rollouts, no training or file writes.

B0 hold; B1 request full sale every decision until filled; B2 SL 5%; B3 TP 10%; B4 either; B5 trailing 5% from observed peak close; B6 at steps 14/29/44/59 actions 1/2/2/4; B7 seeded uniform random. B2–B5 re-evaluate signals each day (no persistent pending orders). B6 skips cancelled scheduled orders, no catch-up, final action is voluntary. Custom interval changes schedule; final decision always full sale. Random generator resets each episode; use distinct reproducible episode seeds for later multi-seed evaluation. Thresholds are illustrative, not optimized. Price rules use raw observed adjusted close, not standardized features; this is causal but not strictly identical observation information to RL. No changes to accounting, reward, or action space. No claim of real-world dividend/share-count accuracy.
