# STEP 6-1: experiment plan (no training)
Install PyYAML if missing: `python -m pip install PyYAML==6.0.3`.
Run from project root:
```
python -m unittest discover -s tests -p test_experiment_configuration.py -v
python -m unittest discover -s tests -p "test_*.py" -v
python -m scripts.prepare_experiment --profile pilot --run-id pilot61
python -m scripts.prepare_experiment --profile full_training --run-id full61
```
Outputs: artifacts/experiment_plans/<run-id>/plan.json. Existing IDs are refused; choose a new ID. Pilot: 6 jobs, 122880 steps; full candidate: 30 jobs, 3072000 steps. Both are plans only. Budgets are aligned to PPO/A2C rollout and DQN train frequency. Checkpoints are intended for one continuous training schedule; do not restart schedules at each checkpoint. Full profile is provisional until pilot analysis. Environment and smoke hyperparameters are deliberately frozen in schema v1; changing them requires a reviewed schema change. Seed/budget/checkpoint settings are configurable within guards.
Input hashes cover manifest, scaler and eight feature files, not raw data. Manifest counts are checked, not full window or scaler semantics. No parquet decoding or model training. Git status and package versions are recorded, not enforced. Plan hash covers resolved configuration and input snapshot, not runtime or source code. STEP 6-2 must recheck input hashes immediately before/after training and record actual runtime, code identity and effective SB3 defaults; a plan alone cannot prove training-time identity. No test performance evaluation. Manifest metadata and whole-file hashes include test-period bytes. Resume and inference replay are future work. Do not commit generated plans containing local runtime details unless reviewed.
