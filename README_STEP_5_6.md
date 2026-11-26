# STEP 5-6 — Validation integrity audit
Copy the four files into the project root, retaining paths. Existing dependencies pandas, numpy, parquet engine required. No training or model inference. Run:

    python -m unittest discover -s tests -p test_integrity_report.py -v
    python -m unittest discover -s tests -p "test_*.py" -v
    python -m scripts.audit_evaluation --source-run validation54 --diagnostic-run diagnostics55 --run-id integrity56

Output: artifacts/evaluation/integrity56/audit.json. Existing output is never overwritten. On failure, preserve source artifacts and inspect traceback. Expected 138 tests if previous 135 unchanged. Checks source hashes, input hashes, diagnostic tables, manifest membership, dates, tradability, next-open cash/cost and next-close value, reward and episode metrics. Fixed pilot assumptions: 100 shares, horizon 60, commission .001, tax .002, slippage .001, lambda .5. No model replay, no proof of training-time parquet identity, no statistical inference. Local parquet audit is required; build environment lacked parquet engine. Three helper tests are not comprehensive end-to-end tests.
