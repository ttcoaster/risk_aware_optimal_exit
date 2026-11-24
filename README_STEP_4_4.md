# STEP 4-4
Extract into existing project root. No existing files replaced.
Run `python -m unittest discover -s tests -p test_finite_horizon.py -v`, then full discovery.
Run `python -m scripts.train_agent_smoke --algorithm all --reward both --steps 2048 --run-id smoke01`.
Use a new alphanumeric run-id for every attempt. Outputs under artifacts/agent_smoke. Do not commit model binaries by default.
Costs are illustrative: commission .001, tax .002, slippage .001. Gamma=1, lambda=.5. Fixed first 10 validation records are integration checks, not representative performance estimates. No test episodes used. FiniteHorizon assumes all underlying truncations are E1 economic terminal horizons. Adjusted price accounting remains a research limitation.
