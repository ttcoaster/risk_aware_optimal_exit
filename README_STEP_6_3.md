# STEP 6-3 checkpoint integration
New files only. Keep the deepcopy fix in STEP 6-2. Run from project root.

`python -m unittest discover -s tests -p test_checkpoint_tracking.py -v`

`python -m scripts.run_tracked_training --plan-id pilot61 --run-id tracking63 --integration --jobs PPO_R1_seed42 PPO_R2_seed42 A2C_R1_seed42 A2C_R2_seed42 DQN_R1_seed42 DQN_R2_seed42`

Expected: 6 jobs, 12288 transitions, 12 checkpoints. Integration saves at 1024 and 2048, not the research schedule. One continuous learn call per job. Capture at next rollout start (after preceding optimizer updates), final capture after learn returns. Checkpoint reload compares policy state_dict hash and step count. Final model also retains STEP 6-2 action probes. No validation/test evaluation, no exact resume, no replay buffer or RNG/environment snapshot. No MLflow. Separate tracked engine intentionally duplicates STEP 6-2 implementation to avoid overwriting local fixes; consolidate after integration verification. Unit tests cover bookkeeping guards only; actual SB3 integration must be run locally.
