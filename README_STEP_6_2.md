# STEP 6-2: plan-bound training engine
Add four files; no replacements. Run from project root in optimal-risk-exit.

    python -m unittest discover -s tests -p test_training_engine.py -v
    python -m unittest discover -s tests -p "test_*.py" -v
    python -m scripts.run_training_engine --plan-id pilot61 --run-id engine62 --integration --jobs PPO_R1_seed42 PPO_R2_seed42 A2C_R1_seed42 A2C_R2_seed42 DQN_R1_seed42 DQN_R2_seed42

Expected 9 new tests, 157 total, 6 jobs and 12288 transitions. Output artifacts/training_engine/engine62. Existing output directories are never overwritten. Use a new run ID after failure. Each job has start.json, model.zip, progress.csv, train.monitor.csv, summary.json. Root summary.json only after all jobs succeed. No validation/test inference. Train probes are separate from monitor. The 2048-step override is integration-only, not a pilot checkpoint. Full training, checkpoint scheduling, exact resume, full replay, and source snapshot verification are not implemented here. Input hashes are checked before/after, not continuously. Runtime records package versions and Git status, not a complete source archive. Algorithm kwargs record explicit settings; SB3 defaults are version dependent. Local integration required: author environment lacks SB3/Gymnasium/Parquet backend. Do not bypass hash failures; restore original inputs or prepare a reviewed new plan.
